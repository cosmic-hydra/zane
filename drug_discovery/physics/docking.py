"""
Molecular Docking Engine
Supports AutoDock Vina and other docking methods
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class DockingEngine:
    """
    Molecular docking for protein-ligand binding prediction
    """

    def __init__(self, method: str = "vina", exhaustiveness: int = 8, num_modes: int = 9):  # 'vina' or 'smina'
        """
        Args:
            method: Docking method to use
            exhaustiveness: Exhaustiveness of search
            num_modes: Number of binding modes to generate
        """
        self.method = method
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

    def dock_ligand(
        self,
        ligand_smiles: str,
        protein_pdb: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    ) -> dict:
        """
        Dock a ligand to a protein

        Args:
            ligand_smiles: Ligand SMILES string
            protein_pdb: Path to protein PDB file or PDB string
            center: Center of docking box (x, y, z)
            box_size: Size of docking box

        Returns:
            Dictionary with docking results
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # Convert SMILES to 3D structure
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                return {"success": False, "error": "Invalid SMILES"}

            mol = Chem.AddHs(mol)
            embed_molecule = getattr(AllChem, "EmbedMolecule")
            success = embed_molecule(mol, randomSeed=42)
            if success == -1:
                return {"success": False, "error": "Failed to generate 3D conformer"}

            mmff_optimize_molecule = getattr(AllChem, "MMFFOptimizeMolecule")
            mmff_optimize_molecule(mol)

            # Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                ligand_pdb = os.path.join(tmpdir, "ligand.pdb")
                ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
                protein_pdbqt = os.path.join(tmpdir, "protein.pdbqt")
                output_pdbqt = os.path.join(tmpdir, "output.pdbqt")

                # Write ligand PDB
                Chem.MolToPDBFile(mol, ligand_pdb)

                # Prepare ligand (convert to PDBQT)
                self._prepare_ligand(ligand_pdb, ligand_pdbqt)

                # Prepare protein (convert to PDBQT)
                if os.path.exists(protein_pdb):
                    self._prepare_protein(protein_pdb, protein_pdbqt)
                else:
                    # Assume it's a PDB string, write to file
                    protein_pdb_file = os.path.join(tmpdir, "protein.pdb")
                    with open(protein_pdb_file, "w") as f:
                        f.write(protein_pdb)
                    self._prepare_protein(protein_pdb_file, protein_pdbqt)

                # Run docking
                scores = self._run_vina_docking(protein_pdbqt, ligand_pdbqt, output_pdbqt, center, box_size)

                if scores:
                    return {
                        "success": True,
                        "binding_affinity": scores[0],  # Best score
                        "all_scores": scores,
                        "num_poses": len(scores),
                    }
                else:
                    return {"success": False, "error": "Docking failed"}

        except Exception as e:
            logger.error(f"Docking error: {e}")
            return {"success": False, "error": str(e)}

    def _prepare_ligand(self, pdb_file: str, pdbqt_file: str):
        """Prepare ligand for docking (PDB to PDBQT)"""
        try:
            # Use Open Babel or obabel if available
            cmd = f"obabel {pdb_file} -O {pdbqt_file} -h"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except Exception as e:
            logger.warning(f"Could not prepare ligand: {e}")
            # Fallback: simple conversion without hydrogen addition
            logger.info("Using fallback ligand preparation")

    def _prepare_protein(self, pdb_file: str, pdbqt_file: str):
        """Prepare protein for docking (PDB to PDBQT)"""
        try:
            # Use prepare_receptor4.py from AutoDockTools
            cmd = f"prepare_receptor4 -r {pdb_file} -o {pdbqt_file}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except Exception as e:
            logger.warning(f"Could not prepare protein: {e}")
            # Fallback: Use obabel
            try:
                cmd = f"obabel {pdb_file} -O {pdbqt_file}"
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
            except Exception:
                logger.error("Protein preparation failed")

    def _run_vina_docking(
        self,
        protein_pdbqt: str,
        ligand_pdbqt: str,
        output_pdbqt: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float],
    ) -> list[float]:
        """Run AutoDock Vina docking"""
        try:
            # Build Vina command
            cmd = [
                "vina",
                "--receptor",
                protein_pdbqt,
                "--ligand",
                ligand_pdbqt,
                "--out",
                output_pdbqt,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(box_size[0]),
                "--size_y",
                str(box_size[1]),
                "--size_z",
                str(box_size[2]),
                "--exhaustiveness",
                str(self.exhaustiveness),
                "--num_modes",
                str(self.num_modes),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse output for binding affinities
            scores = []
            for line in result.stdout.split("\n"):
                if line.strip().startswith("1") and "kcal/mol" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1])
                            scores.append(score)
                        except ValueError:
                            continue

            return scores

        except FileNotFoundError:
            logger.error("AutoDock Vina not found. Please install it.")
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"Vina docking failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Docking error: {e}")
            return []

    def batch_dock(
        self,
        ligand_smiles_list: list[str],
        protein_pdb: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    ) -> list[dict]:
        """
        Dock multiple ligands to a protein

        Args:
            ligand_smiles_list: List of ligand SMILES
            protein_pdb: Path to protein PDB
            center: Docking box center
            box_size: Docking box size

        Returns:
            List of docking results
        """
        results = []

        for smiles in ligand_smiles_list:
            result = self.dock_ligand(smiles, protein_pdb, center, box_size)
            results.append(result)

        return results

    def virtual_screening(
        self,
        ligand_smiles_list: list[str],
        protein_pdb: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
        score_threshold: float = -7.0,
    ) -> list[tuple[str, float]]:
        """
        Perform virtual screening on a library of compounds

        Args:
            ligand_smiles_list: Library of ligands
            protein_pdb: Target protein
            center: Docking box center
            box_size: Docking box size
            score_threshold: Minimum binding affinity threshold

        Returns:
            List of (SMILES, score) tuples for hits
        """
        hits = []

        results = self.batch_dock(ligand_smiles_list, protein_pdb, center, box_size)

        for smiles, result in zip(ligand_smiles_list, results):
            if result.get("success") and result.get("binding_affinity"):
                score = result["binding_affinity"]
                if score <= score_threshold:  # More negative = better binding
                    hits.append((smiles, score))

        # Sort by binding affinity (most negative first)
        hits.sort(key=lambda x: x[1])

        logger.info(f"Virtual screening: {len(hits)} hits found from {len(ligand_smiles_list)} compounds")

        return hits
