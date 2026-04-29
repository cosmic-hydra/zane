try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

from rdkit import Chem

def parse_cif_to_mol(cif_path: str) -> Chem.Mol:
    if not GEMMI_AVAILABLE:
        raise ImportError(&quot;gemmi required for CIF parsing&quot;)
    structure = gemmi.read_structure(cif_path)
    # Convert first model/chain to PDB string
    pdb_block = structure.make_pdb_block().as_string()
    mol = Chem.MolFromPDBBlock(pdb_block)
    return mol