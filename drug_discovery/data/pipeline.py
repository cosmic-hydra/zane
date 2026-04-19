"""
Molecular Data Pipeline for ZANE.

Production-grade molecular data processing:
- SMILES validation and canonicalization
- Physicochemical descriptors (RDKit)
- Morgan fingerprints + Tanimoto similarity
- 3D graph construction for GNNs
- Lipinski Rule of Five filtering
- Batch quality reporting
"""

from __future__ import annotations
import logging, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def is_valid_smiles_fast(smiles):
    if not smiles or len(smiles) > 2000: return False
    return smiles.count("(") == smiles.count(")") and smiles.count("[") == smiles.count("]")

def validate_smiles(smiles):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return (True, Chem.MolToSmiles(mol, isomericSmiles=True)) if mol else (False, None)
    except ImportError:
        return is_valid_smiles_fast(smiles), smiles

def validate_batch(smiles_list):
    valid, invalid, canonical = [], [], []
    for s in smiles_list:
        ok, c = validate_smiles(s)
        if ok: valid.append(s); canonical.append(c)
        else: invalid.append(s)
    uc = list(set(canonical))
    return {"total": len(smiles_list), "valid": len(valid), "invalid": len(invalid),
            "valid_ratio": len(valid)/max(len(smiles_list),1), "unique_canonical": len(uc),
            "duplicate_count": len(canonical)-len(uc), "invalid_smiles": invalid[:20]}

def compute_descriptors(smiles):
    try:
        from rdkit import Chem; from rdkit.Chem import Descriptors, Crippen
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return {"mol_weight": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol), "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol), "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol), "qed": Descriptors.qed(mol)}
    except ImportError: return None

def lipinski_filter(desc):
    checks = {"mw_le_500": desc.get("mol_weight",0)<=500, "logp_le_5": desc.get("logp",0)<=5,
              "hbd_le_5": desc.get("hbd",0)<=5, "hba_le_10": desc.get("hba",0)<=10}
    v = sum(1 for c in checks.values() if not c)
    return {"passes": v<=1, "violations": v, "checks": checks}

def compute_morgan_fingerprint(smiles, radius=2, nbits=2048):
    try:
        from rdkit import Chem; from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), dtype=np.float32)
    except ImportError: return None

def tanimoto_similarity(fp1, fp2):
    inter = np.sum(fp1*fp2); union = np.sum(fp1)+np.sum(fp2)-inter
    return float(inter / max(union, 1e-12))

def smiles_to_graph(smiles):
    try:
        from rdkit import Chem; from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG()); AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            conf = mol.GetConformer()
            pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        except: pos = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        src, dst = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); src.extend([i,j]); dst.extend([j,i])
        ei = np.array([src, dst], dtype=np.int64) if src else np.zeros((2,0), dtype=np.int64)
        return {"z": z, "pos": pos, "edge_index": ei, "num_atoms": len(z)}
    except ImportError: return None

@dataclass
class MolecularDataset:
    smiles: List[str] = field(default_factory=list)
    targets: Optional[np.ndarray] = None
    descriptors: Optional[List[Dict]] = None
    fingerprints: Optional[np.ndarray] = None
    graphs: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self): return len(self.smiles)

    def featurize(self, methods=None):
        methods = methods or ["descriptors", "fingerprints"]
        if "descriptors" in methods:
            self.descriptors = [compute_descriptors(s) for s in self.smiles]
            logger.info(f"Computed descriptors for {self.size} molecules")
        if "fingerprints" in methods:
            fps = [compute_morgan_fingerprint(s) for s in self.smiles]
            valid = [f for f in fps if f is not None]
            self.fingerprints = np.array(valid) if valid else None
        if "graphs" in methods:
            self.graphs = [smiles_to_graph(s) for s in self.smiles]

    def quality_report(self):
        report = validate_batch(self.smiles)
        if self.descriptors:
            vd = [d for d in self.descriptors if d]
            if vd:
                report["lipinski_pass_rate"] = sum(1 for d in vd if lipinski_filter(d)["passes"])/len(vd)
                report["mean_qed"] = np.mean([d["qed"] for d in vd if "qed" in d])
        return report
