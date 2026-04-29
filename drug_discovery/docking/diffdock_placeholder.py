def run_diffdock(ligand_smiles: str, protein_pdb: str, num_poses: int = 10) -> dict:
    # Stub for external/DiffDock integration
    # In full: subprocess.call(['python', 'external/DiffDock/run.py', ...])
    return {
        'poses': [], 
        'affinity': -7.5,  # kcal/mol stub
        'rmsd': 1.2,
        'success': True
    }