import ray

from drug_discovery.geometric_dl.fep_engine import FEPEngine
from drug_discovery.simulation.free_energy import compute_binding_free_energy


@ray.remote(num_gpus=1)
def process_smiles_batch(smiles_list):
    """
    Process a batch of SMILES strings for 4D geometric physics calculations.
    """
    results = []
    # Initialize engine within the worker
    _ = FEPEngine()

    for smiles in smiles_list:
        try:
            # Perform heavy geometric physics calculations
            dg_metric = compute_binding_free_energy(smiles)
            results.append({"smiles": smiles, "dg": dg_metric, "status": "success"})
        except Exception as e:
            results.append({"smiles": smiles, "error": str(e), "status": "failed"})
    return results


def run_distributed_simulations(smiles_source, batch_size=100):
    ray.init(address="auto")

    # Split SMILES into batches
    batches = [smiles_source[i : i + batch_size] for i in range(0, len(smiles_source), batch_size)]

    # Distribute work across the cluster
    futures = [process_smiles_batch.remote(batch) for batch in batches]

    # Collect results
    results = ray.get(futures)
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    # Example usage
    sample_smiles = ["CCO", "CCN", "CCC"] * 33333  # Approx 100k
    print(f"Starting distributed simulation for {len(sample_smiles)} molecules...")
    # results = run_distributed_simulations(sample_smiles)
    # print(f"Processed {len(results)} molecules.")
