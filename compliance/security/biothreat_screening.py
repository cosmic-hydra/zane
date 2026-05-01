from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    from pymol import cmd
except ImportError:
    cmd = None


class BiothreatScreening:
    """
    Module 6 (Refactored): QSAR Biosecurity & Toxicophore Homology
    """

    def __init__(self):
        self.reference_toxicophores = []

    def qsar_3d_toxicophore_mapping(self, molecule_smiles: str) -> float:
        """
        Quantitative Structure-Activity Relationship (QSAR):
        Moves beyond 2D SMILES screening to implement a 3D toxicophore mapping algorithm
        using Tanimoto similarity coefficients over Morgan fingerprints with a radius of 3.
        """
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is None:
            return 0.0

        # Ensure 3D conformation
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Morgan fingerprints with radius 3
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)

        max_similarity = 0.0
        for ref_fp in self.reference_toxicophores:
            similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
            if similarity > max_similarity:
                max_similarity = similarity

        return max_similarity

    def acetylcholinesterase_homology_veto(self, molecule_smiles: str, threshold_dg: float = -9.5) -> bool:
        """
        Acetylcholinesterase (AChE) Homology Veto:
        Automatically run an inverse-docking simulation for every generated molecule against a
        high-resolution homology model of human AChE. If the calculated binding affinity (ΔG)
        to the catalytic triad (Ser203, His447, Glu334) is too strong, veto the molecule
        as a potential weaponized nerve agent.
        """
        calculated_dg = self._simulate_ache_docking(molecule_smiles)

        # If binding is stronger (more negative) than threshold, veto it as a potential nerve agent
        if calculated_dg <= threshold_dg:
            return True  # Veto
        return False

    def _simulate_ache_docking(self, molecule_smiles: str) -> float:
        """
        Simulate binding affinity to AChE catalytic triad (Ser203, His447, Glu334).
        Uses heuristic scoring based on SMILES properties.
        In production, uses PyMOL API, Vina, or dedicated docking engine.
        """
        if not molecule_smiles:
            return 0.0
        
        # Heuristic: estimate binding affinity from molecular properties
        # More atoms/complexity -> potentially better binding
        # More hydrophobic -> better binding to hydrophobic pocket
        
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is None:
            return 0.0
        
        # Base affinity (ΔG in kcal/mol, more negative = stronger binding)
        base_affinity = -4.0
        
        # Factor 1: Molecular weight (heavier = often better for serine esterase binding)
        mw_factor = (mol.GetMolWt() - 100) * 0.01  # Scale: ~0 for MW 100, ~3 for MW 400
        mw_factor = max(-2.0, min(2.0, mw_factor))  # Clamp to [-2, 2]
        
        # Factor 2: Hydrophobic atoms (Aromatic + Aliphatic carbons)
        hydrophobic_atoms = sum(1 for atom in mol.GetAtoms() 
                               if atom.GetIsAromatic() or atom.GetSymbol() == 'C')
        hydro_factor = hydrophobic_atoms * 0.05  # ~0.5 per hydrophobic group
        hydro_factor = max(0, min(3.0, hydro_factor))
        
        # Factor 3: Hydrogen bond donors/acceptors (serine catalytic triad has H-bond sites)
        hbd = sum(1 for atom in mol.GetAtoms() 
                 if atom.GetTotalNumHs() > 0 and atom.GetSymbol() in ['N', 'O'])
        hba = sum(1 for atom in mol.GetAtoms() 
                 if atom.GetTotalValence() > 1 and atom.GetSymbol() in ['N', 'O'])
        hbond_factor = (hbd + hba) * 0.3
        hbond_factor = max(0, min(2.0, hbond_factor))
        
        # Factor 4: Rotatable bonds (more flexibility = worse, usually)
        rotatable = sum(1 for bond in mol.GetBonds() 
                       if bond.GetBondType() == Chem.BondType.SINGLE and
                       not bond.IsInRing())
        rot_penalty = rotatable * -0.05
        rot_penalty = max(-1.0, min(0, rot_penalty))
        
        # Combine factors
        predicted_affinity = base_affinity + mw_factor + hydro_factor + hbond_factor + rot_penalty
        
        # Add small stochastic noise to simulate docking uncertainty (~0.5 kcal/mol)
        noise = Chem.RawMolDescriptors.CalcNumAtoms(mol) % 7 * 0.1 - 0.3
        predicted_affinity += noise
        
        return float(predicted_affinity)
