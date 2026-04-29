from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
import os

class VinaDocker:
    def __init__(self, exhaustiveness: int = 8):
        self.vina = Vina(sf_name='vina')
        self.exhaustiveness = exhaustiveness

    def prepare_receptor(self, pdb_path: str) -> str:
        # Simple prep: assume clean PDB
        receptor_pdbqt = pdb_path.replace('.pdb', '_receptor.pdbqt')
        os.system(f'obabel {pdb_path} -O {receptor_pdbqt} -xh')  # Add H
        return receptor_pdbqt

    def prepare_ligands(self, sdf_path: str) -> str:
        ligands_pdbqt = sdf_path.replace('.sdf', '_ligands.pdbqt')
        os.system(f'obabel {sdf_path} -O {ligands_pdbqt} -xh')
        return ligands_pdbqt

    def dock(self, receptor_pdb: str, ligands_sdf: str, center: tuple = (0,0,0), size: tuple = (20,20,20)) -> dict:
        receptor = self.prepare_receptor(receptor_pdb)
        ligands = self.prepare_ligands(ligands_sdf)
        self.vina.set_receptor(receptor)
        self.vina.set_ligands(ligands)
        self.vina.compute_vina_maps(center=center, box_size=list(size))
        self.vina.dock(exhaustiveness=self.exhaustiveness, n_poses=9)
        affinities = self.vina.energies()
        return {'affinities': affinities, 'poses': self.vina.poses()}