#include <torch/extension.h>
#include <stdexcept>
#include <vector>

namespace {

torch::Tensor ensure_batch(torch::Tensor coords) {
    if (coords.dim() == 2) {
        return coords.unsqueeze(0);
    }
    if (coords.dim() != 3) {
        throw std::invalid_argument("coords must be (N,3) or (B,N,3)");
    }
    return coords;
}

}  // namespace

torch::Tensor compute_energy(torch::Tensor coords, double sigma, double epsilon) {
    auto c = ensure_batch(coords);
    auto dist = torch::cdist(c, c) + 1e-9;
    auto eye = torch::eye(dist.size(-1), dist.options()).unsqueeze(0);
    dist = dist + eye * 1e6;

    auto inv = torch::pow((sigma / dist), 6);
    auto energy_mat = 4.0 * epsilon * (inv * inv - inv);
    auto energy = energy_mat.triu(1).sum(-1).sum(-1);
    return energy;
}

torch::Tensor compute_forces(torch::Tensor coords, double sigma, double epsilon) {
    auto c = ensure_batch(coords);
    auto diff = c.unsqueeze(2) - c.unsqueeze(1);  // (B, N, N, 3)
    auto dist2 = (diff * diff).sum(-1) + 1e-9;
    const auto n_atoms = diff.size(1);
    auto mask = torch::eye(n_atoms, dist2.options()).unsqueeze(0);
    dist2 = dist2 + mask * 1e6;

    auto inv2 = (sigma * sigma) / dist2;
    auto inv6 = inv2 * inv2 * inv2;
    auto coeff = 24.0 * epsilon * (2 * inv6 * inv6 - inv6) / dist2;
    coeff = coeff * (1.0 - mask);
    auto forces = (coeff.unsqueeze(-1) * diff).sum(2);
    return forces;
}

torch::Tensor run_fep(torch::Tensor ligand, torch::Tensor protein, torch::Tensor lambda_schedule, double sigma, double epsilon) {
    auto l = ensure_batch(ligand);
    auto p = ensure_batch(protein);
    auto device = l.device();

    torch::Tensor lambdas = lambda_schedule;
    if (!lambda_schedule.defined() || lambda_schedule.numel() == 0) {
        lambdas = torch::linspace(0.0, 1.0, 5, torch::TensorOptions().device(device).dtype(l.scalar_type()));
    }
    if (lambdas.dim() == 0) {
        lambdas = lambdas.unsqueeze(0);
    }

    auto cross = torch::cdist(l, p) + 1e-9;
    auto inv = torch::pow((sigma / cross), 6);
    auto interaction = 4.0 * epsilon * (inv * inv - inv);  // (B, n_lig, n_prot)

    auto delta_f = torch::zeros({l.size(0)}, torch::TensorOptions().device(device).dtype(l.scalar_type()));
    for (int64_t i = 0; i < lambdas.size(0) - 1; ++i) {
        auto lam_left = lambdas[i];
        auto lam_right = lambdas[i + 1];
        auto lam_mid = 0.5 * (lam_left + lam_right);
        auto scaled = interaction * lam_mid;
        auto slice_energy = scaled.sum(-1).sum(-1);
        delta_f += (lam_right - lam_left) * slice_energy;
    }
    return delta_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_energy", &compute_energy, "Compute Lennard-Jones energy (supports CUDA)");
    m.def("compute_forces", &compute_forces, "Compute forces for each atom");
    m.def("run_fep", &run_fep, "Free-energy perturbation estimation");
}
