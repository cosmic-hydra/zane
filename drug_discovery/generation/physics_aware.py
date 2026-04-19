"""
Physics-aware, risk-aware, and synthesis-aware molecular generation pipeline.

Implements fragment-first design, conformer ensemble diffusion, energy landscape
analysis, and multi-objective scoring that couples docking realism, synthesis
feasibility, and safety constraints.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from drug_discovery.physics.md_simulator import MolecularDynamicsSimulator
from drug_discovery.synthesis.retrosynthesis import RetrosynthesisPlanner, SynthesisFeasibilityScorer

try:  # Optional heavy deps are guarded to keep CLI startup light.
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, BRICS, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    np = None


DEFAULT_FRAGMENT_SEEDS = [
    "c1ccccc1",
    "c1nccc2cc(O)ccc12",
    "C1CCC(N)CC1",
    "CC(=O)O",
    "c1ccncc1",
    "n1cccc1",
    "COc1ccccc1",
    "c1ccc(S(=O)(=O)N)cc1",
]


@dataclass
class FragmentAssembly:
    smiles: str
    fragments_used: list[str]
    fragment_compatibility: float
    steric_fit: float


@dataclass
class ConformerSnapshot:
    conformer_id: int
    energy: float
    boltzmann_weight: float
    rmsd_to_lowest: float


@dataclass
class ConformerEnsemble:
    lowest_energy: float
    boltzmann_average: float
    snapshots: list[ConformerSnapshot] = field(default_factory=list)
    diffusion_trace: list[float] = field(default_factory=list)


@dataclass
class RiskProfile:
    total_risk: float
    toxicity: float
    reactivity: float
    synthetic_difficulty: float
    alerts: list[str] = field(default_factory=list)


@dataclass
class CandidateAssessment:
    smiles: str
    fragments: list[str]
    fragment_score: float
    steric_fit: float
    ensemble: ConformerEnsemble
    risk: RiskProfile
    reaction: dict[str, Any]
    pharmacophore: dict[str, Any]
    scaffold_hops: list[str]
    quantum: dict[str, float]
    md: dict[str, Any]
    scores: dict[str, float]


class PhysicsAwareGenerator:
    """
    Physics-aware generator that assembles BRICS/RECAP fragments, explores
    conformer ensembles, and ranks molecules with multi-objective, risk-aware
    scoring. Designed to produce low-risk, synthesizable candidates while
    maintaining docking realism.
    """

    def __init__(
        self,
        exploration_temperature: float = 0.7,
        max_conformers: int = 8,
        max_candidates: int = 20,
        target_temperature_k: float = 300.0,
    ):
        self.exploration_temperature = max(0.2, min(1.5, exploration_temperature))
        self.max_conformers = max(3, max_conformers)
        self.max_candidates = max(1, max_candidates)
        self.simulator = MolecularDynamicsSimulator(temperature=target_temperature_k)
        self.retro = RetrosynthesisPlanner()
        self.synthesis_scorer = SynthesisFeasibilityScorer()
        self.rng = random.Random(42)

    def _canonicalize(self, smiles: str) -> str | None:
        if Chem is None:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                return None
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return None

    def _fragment_library(self, seeds: Sequence[str] | None) -> list[str]:
        if Chem is None:
            return list(DEFAULT_FRAGMENT_SEEDS)
        fragments: set[str] = set()
        for smi in seeds or DEFAULT_FRAGMENT_SEEDS:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                fragments.update(BRICS.BRICSDecompose(mol, silent=True))
            except Exception:
                pass
            try:
                recap = rdMolDescriptors.RecapDecompose(mol)
                if recap:
                    fragments.update(node.smiles for node in recap.GetAllChildren().values())
            except Exception:
                pass
        clean = [f for f in fragments if f and len(f) >= 2]
        if not clean:
            clean = list(DEFAULT_FRAGMENT_SEEDS)
        self.rng.shuffle(clean)
        return clean[: max(40, self.max_candidates * 2)]

    def _assemble_fragments(self, fragments: Sequence[str], target_pocket: dict[str, float] | None = None) -> list[FragmentAssembly]:
        if Chem is None:
            return [
                FragmentAssembly(smiles=smi, fragments_used=[smi], fragment_compatibility=0.4, steric_fit=0.4)
                for smi in fragments[: self.max_candidates]
            ]
        mol_frags = [Chem.MolFromSmiles(f) for f in fragments if Chem.MolFromSmiles(f)]
        assemblies: list[FragmentAssembly] = []
        try:
            builder = BRICS.BRICSBuild(mol_frags)
            for mol in builder:
                smi = Chem.MolToSmiles(mol, canonical=True)
                if not smi:
                    continue
                comp = self._fragment_compatibility_score(mol_frags)
                steric = self._steric_fit_score(mol, target_pocket)
                assemblies.append(FragmentAssembly(smiles=smi, fragments_used=[], fragment_compatibility=comp, steric_fit=steric))
                if len(assemblies) >= self.max_candidates * 3:
                    break
        except Exception:
            pass
        if not assemblies:
            for smi in fragments[: self.max_candidates]:
                assemblies.append(
                    FragmentAssembly(smiles=smi, fragments_used=[smi], fragment_compatibility=0.35, steric_fit=0.35)
                )
        return assemblies

    def _fragment_compatibility_score(self, mol_frags: Sequence[Any]) -> float:
        if not mol_frags or Chem is None:
            return 0.4
        sizes = [frag.GetNumHeavyAtoms() for frag in mol_frags if frag]
        if not sizes:
            return 0.4
        spread = max(sizes) - min(sizes)
        balance = 1.0 - min(spread / max(1, max(sizes)), 1.0)
        return round(0.5 + 0.5 * balance, 3)

    def _steric_fit_score(self, mol: Any, target_pocket: dict[str, float] | None) -> float:
        if Chem is None:
            return 0.4
        heavy = mol.GetNumHeavyAtoms()
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        pocket_size = float(target_pocket.get("volume", 450.0)) if target_pocket else 450.0
        bulk_penalty = min(1.0, max(0.0, (heavy * 12.0 - pocket_size) / pocket_size))
        flexibility_penalty = min(1.0, rot / 15.0)
        fit = max(0.0, 1.0 - 0.6 * bulk_penalty - 0.2 * flexibility_penalty)
        return round(fit, 3)

    def _conformer_ensemble(self, smiles: str) -> ConformerEnsemble:
        if Chem is None or np is None:
            return ConformerEnsemble(lowest_energy=0.0, boltzmann_average=0.0, snapshots=[], diffusion_trace=[])
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            return ConformerEnsemble(lowest_energy=0.0, boltzmann_average=0.0, snapshots=[], diffusion_trace=[])
        try:
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.max_conformers, params=AllChem.ETKDGv3())
        except Exception:
            ids = []
        energies: list[float] = []
        snapshots: list[ConformerSnapshot] = []
        diffusion_trace: list[float] = []
        for cid in ids:
            try:
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                ff.Minimize()
                energy = float(ff.CalcEnergy())
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol, confId=cid)
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                    energy = float(ff.CalcEnergy())
                except Exception:
                    energy = 0.0
            energies.append(energy)
            diffusion_trace.append(energy)
        if not energies:
            return ConformerEnsemble(lowest_energy=0.0, boltzmann_average=0.0, snapshots=[], diffusion_trace=[])
        min_e = min(energies)
        k_b = 0.0019872041  # kcal/(mol*K)
        t = 300.0
        weights = [math.exp(-(e - min_e) / (k_b * t)) for e in energies]
        weight_sum = sum(weights) or 1.0
        boltz_avg = sum(e * w for e, w in zip(energies, weights)) / weight_sum
        ref_conf = int(energies.index(min_e))
        ref_coords = mol.GetConformer(ref_conf).GetPositions()
        for idx, energy in enumerate(energies):
            coords = mol.GetConformer(idx).GetPositions()
            rmsd = float(np.sqrt(((coords - ref_coords) ** 2).sum(axis=1).mean()))
            snapshots.append(
                ConformerSnapshot(
                    conformer_id=idx,
                    energy=round(energy, 4),
                    boltzmann_weight=round(weights[idx] / weight_sum, 4),
                    rmsd_to_lowest=round(rmsd, 3),
                )
            )
        return ConformerEnsemble(
            lowest_energy=round(min_e, 4),
            boltzmann_average=round(boltz_avg, 4),
            snapshots=sorted(snapshots, key=lambda s: s.energy),
            diffusion_trace=[round(e, 4) for e in diffusion_trace],
        )

    def _risk_profile(self, smiles: str) -> RiskProfile:
        alerts: list[str] = []
        if Chem is None:
            return RiskProfile(total_risk=0.5, toxicity=0.2, reactivity=0.15, synthetic_difficulty=0.15, alerts=alerts)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return RiskProfile(total_risk=1.0, toxicity=0.5, reactivity=0.3, synthetic_difficulty=0.2, alerts=["invalid_smiles"])
        alert_smarts = {
            "nitro": "[N+](=O)[O-]",
            "aniline": "Nc1ccc(cc1)",
            "aldehyde": "[CX3H1](=O)[#6]",
            "epoxide": "[OX2r3]",
            "aziridine": "[NX3r3]",
            "quaternary_ammonium": "[N+](C)(C)(C)C",
        }
        tox_score = 0.1
        for name, pattern in alert_smarts.items():
            patt = Chem.MolFromSmarts(pattern)
            if patt and mol.HasSubstructMatch(patt):
                alerts.append(name)
                tox_score += 0.15
        logp = Descriptors.MolLogP(mol)
        tox_score += max(0.0, (abs(logp - 2.5) - 1.5) / 10.0)
        reactivity = min(1.0, 0.1 * rdMolDescriptors.CalcNumRotatableBonds(mol) + 0.05 * Lipinski.NumHBD(mol))
        sa = self.synthesis_scorer.score_feasibility(smiles).get("overall", 5.0)
        synthetic_difficulty = 1.0 - min(1.0, sa / 10.0)
        total = min(1.0, tox_score + 0.5 * reactivity + 0.5 * synthetic_difficulty)
        return RiskProfile(
            total_risk=round(total, 4),
            toxicity=round(min(1.0, tox_score), 4),
            reactivity=round(min(1.0, reactivity), 4),
            synthetic_difficulty=round(min(1.0, synthetic_difficulty), 4),
            alerts=alerts,
        )

    def _reaction_feasibility(self, smiles: str) -> dict[str, Any]:
        plan = self.retro.plan_synthesis(smiles)
        steps = float(plan.get("num_steps", 6))
        est_yield = float(plan.get("estimated_yield", 0.5))
        likelihood = max(0.0, min(1.0, 0.7 * est_yield + 0.3 * (1.0 - steps / 10.0)))
        plan["reaction_likelihood"] = round(likelihood, 4)
        return plan

    def _pharmacophore_profile(self, smiles: str, constraints: dict[str, Any] | None) -> dict[str, Any]:
        if Chem is None:
            return {"hba": 0, "hbd": 0, "aromatic_rings": 0, "satisfied": True}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"hba": 0, "hbd": 0, "aromatic_rings": 0, "satisfied": False}
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        satisfied = True
        if constraints:
            if "min_hba" in constraints:
                satisfied &= hba >= constraints["min_hba"]
            if "min_hbd" in constraints:
                satisfied &= hbd >= constraints["min_hbd"]
            if "max_rings" in constraints:
                satisfied &= rings <= constraints["max_rings"]
        return {"hba": hba, "hbd": hbd, "aromatic_rings": rings, "satisfied": bool(satisfied)}

    def _scaffold_hops(self, smiles: str, fragments: Sequence[str]) -> list[str]:
        if Chem is None:
            return []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            generic_core = MurckoScaffold.MakeScaffoldGeneric(core)
            core_smiles = Chem.MolToSmiles(generic_core, canonical=True)
        except Exception:
            core_smiles = ""
        hops: list[str] = []
        for frag in fragments[:5]:
            try:
                frag_mol = Chem.MolFromSmiles(frag)
                if not frag_mol or not core_smiles:
                    continue
                hop = Chem.ReplaceCore(mol, frag_mol)
                if hop:
                    hops.append(Chem.MolToSmiles(hop, canonical=True))
            except Exception:
                continue
        if not hops and core_smiles:
            hops.append(core_smiles)
        return list(dict.fromkeys(hops))

    def _quantum_descriptors(self, smiles: str) -> dict[str, float]:
        if Chem is None:
            return {"homo_lumo_gap": 0.0, "min_partial_charge": 0.0, "max_partial_charge": 0.0, "esp_range": 0.0}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"homo_lumo_gap": 0.0, "min_partial_charge": 0.0, "max_partial_charge": 0.0, "esp_range": 0.0}
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetDoubleProp("_GasteigerCharge")) for atom in mol.GetAtoms()]
        except Exception:
            charges = [0.0]
        homo_lumo_gap = max(0.0, 5.5 - 0.1 * Lipinski.HeavyAtomCount(mol) + 0.05 * rdMolDescriptors.CalcTPSA(mol))
        return {
            "homo_lumo_gap": round(homo_lumo_gap, 3),
            "min_partial_charge": round(min(charges), 4),
            "max_partial_charge": round(max(charges), 4),
            "esp_range": round(max(charges) - min(charges), 4),
        }

    def _md_feedback(self, smiles: str) -> dict[str, Any]:
        try:
            return self.simulator.simulate_ligand(smiles, num_steps=4000, minimize=True)
        except Exception as exc:  # pragma: no cover - best effort fallback
            return {"success": False, "error": str(exc)}

    def _multi_objective_score(
        self,
        ensemble: ConformerEnsemble,
        risk: RiskProfile,
        reaction: dict[str, Any],
        pharmacophore: dict[str, Any],
        md: dict[str, Any],
    ) -> dict[str, float]:
        binding_proxy = max(0.0, min(1.0, 0.7 * (1.0 - risk.reactivity) + 0.3 * (1.0 - ensemble.boltzmann_average / 50.0)))
        admet = max(0.0, min(1.0, 0.6 * (1.0 - risk.toxicity) + 0.4 * (1.0 - risk.synthetic_difficulty)))
        synth = reaction.get("reaction_likelihood", 0.5)
        novelty = max(0.0, 1.0 - risk.synthetic_difficulty * 0.5 - len(risk.alerts) * 0.05)
        md_stability = float(md.get("stability_index", 0.5)) if md else 0.5
        multi = (
            0.25 * binding_proxy
            + 0.2 * admet
            + 0.2 * synth
            + 0.15 * novelty
            + 0.2 * md_stability
            - 0.2 * risk.total_risk
        )
        uncertainty = abs(ensemble.boltzmann_average - ensemble.lowest_energy) / 50.0
        return {
            "binding_proxy": round(binding_proxy, 4),
            "admet": round(admet, 4),
            "synthesizability": round(synth, 4),
            "novelty": round(novelty, 4),
            "md_stability": round(md_stability, 4),
            "multi_objective": round(max(0.0, min(1.0, multi)), 4),
            "uncertainty": round(min(1.0, uncertainty), 4),
            "constraints_satisfied": 1.0 if pharmacophore.get("satisfied", False) else 0.0,
        }

    def _chemical_space_metrics(self, smiles_list: Sequence[str], known: Sequence[str] | None) -> dict[str, float]:
        if Chem is None or not smiles_list:
            return {"diversity": 0.0, "novelty": 0.0}
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=2, nBits=1024) for s in smiles_list]
        diversity_scores: list[float] = []
        for idx, fp in enumerate(fps):
            others = fps[:idx] + fps[idx + 1 :]
            if not others:
                continue
            sims = BulkTanimotoSimilarity(fp, others)
            diversity_scores.append(1.0 - float(np.mean(sims)))
        novelty_scores: list[float] = []
        if known:
            known_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=2, nBits=1024) for s in known]
            for fp in fps:
                sims = BulkTanimotoSimilarity(fp, known_fps)
                novelty_scores.append(1.0 - float(np.max(sims)))
        return {
            "diversity": round(float(np.mean(diversity_scores) if diversity_scores else 0.0), 4),
            "novelty": round(float(np.mean(novelty_scores) if novelty_scores else 0.0), 4),
        }

    def generate(
        self,
        seed_smiles: Sequence[str] | None = None,
        num_molecules: int = 10,
        target_protein: str | None = None,
        pharmacophore_constraints: dict[str, Any] | None = None,
        known_smiles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        fragments = self._fragment_library(seed_smiles)
        assemblies = self._assemble_fragments(fragments, target_pocket={"volume": 500.0} if target_protein else None)

        candidates: list[CandidateAssessment] = []
        for asm in assemblies:
            canonical = self._canonicalize(asm.smiles)
            if not canonical:
                continue
            ensemble = self._conformer_ensemble(canonical)
            risk = self._risk_profile(canonical)
            if risk.total_risk >= 0.95:
                continue  # prune high-risk regions in latent space
            reaction = self._reaction_feasibility(canonical)
            pharm = self._pharmacophore_profile(canonical, pharmacophore_constraints)
            scaffold_hops = self._scaffold_hops(canonical, fragments)
            quantum = self._quantum_descriptors(canonical)
            md = self._md_feedback(canonical)
            scores = self._multi_objective_score(ensemble, risk, reaction, pharm, md)

            candidates.append(
                CandidateAssessment(
                    smiles=canonical,
                    fragments=[f for f in asm.fragments_used or []] or fragments[:3],
                    fragment_score=asm.fragment_compatibility,
                    steric_fit=asm.steric_fit,
                    ensemble=ensemble,
                    risk=risk,
                    reaction=reaction,
                    pharmacophore=pharm,
                    scaffold_hops=scaffold_hops,
                    quantum=quantum,
                    md=md,
                    scores=scores,
                )
            )

        candidates.sort(key=lambda c: c.scores["multi_objective"], reverse=True)
        selected = candidates[: num_molecules]

        diversity = self._chemical_space_metrics([c.smiles for c in selected], known_smiles)
        experimental_loop = [
            {
                "iteration": 1,
                "generated": len(selected),
                "avg_score": round(float(sum(c.scores["multi_objective"] for c in selected) / max(1, len(selected))), 4),
                "success_rate": round(float(sum(1 for c in selected if c.md.get("stable")) / max(1, len(selected))), 4),
            }
        ]

        return {
            "success": True,
            "num_candidates": len(selected),
            "fragments_used": fragments[:20],
            "candidates": [
                {
                    "smiles": c.smiles,
                    "fragment_score": c.fragment_score,
                    "steric_fit": c.steric_fit,
                    "ensemble": {
                        "lowest_energy": c.ensemble.lowest_energy,
                        "boltzmann_average": c.ensemble.boltzmann_average,
                        "snapshots": [s.__dict__ for s in c.ensemble.snapshots],
                        "diffusion_trace": c.ensemble.diffusion_trace,
                    },
                    "risk": c.risk.__dict__,
                    "reaction": c.reaction,
                    "pharmacophore": c.pharmacophore,
                    "scaffold_hops": c.scaffold_hops,
                    "quantum": c.quantum,
                    "md": c.md,
                    "scores": c.scores,
                }
                for c in selected
            ],
            "chemical_space": diversity,
            "experimental_loop": experimental_loop,
            "objectives": ["binding", "admet", "synthesizability", "novelty", "md_stability"],
        }
