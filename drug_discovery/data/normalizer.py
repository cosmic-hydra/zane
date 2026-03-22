"""
Data Normalizer - SMILES Standardization, Deduplication, Canonicalization

Ensures consistent molecular representations across all data sources.
"""

import logging
from typing import List, Set, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize molecular data for consistency and quality."""

    def __init__(self, remove_salts: bool = True, remove_duplicates: bool = True):
        """
        Initialize the data normalizer.

        Args:
            remove_salts: Remove salt components from molecules
            remove_duplicates: Remove duplicate molecules
        """
        self.remove_salts = remove_salts
        self.remove_duplicates = remove_duplicates
        self._seen_inchikeys: Set[str] = set()

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Remove salts if requested
            if self.remove_salts:
                # Keep largest fragment
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    mol = max(frags, key=lambda m: m.GetNumAtoms())

            canonical = Chem.MolToSmiles(mol, canonical=True)
            return canonical

        except Exception as e:
            logger.warning(f"Failed to canonicalize SMILES '{smiles}': {e}")
            return None

    def compute_inchikey(self, smiles: str) -> Optional[str]:
        """
        Compute InChIKey for molecular uniqueness.

        Args:
            smiles: Input SMILES string

        Returns:
            InChIKey or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except Exception as e:
            logger.warning(f"Failed to compute InChIKey for '{smiles}': {e}")
            return None

    def is_valid_molecule(self, smiles: str) -> bool:
        """
        Check if SMILES represents a valid molecule.

        Args:
            smiles: Input SMILES string

        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Basic filters
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 3 or num_atoms > 150:  # Reasonable size range
                return False

            # Check for valid valence
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return False

            return True

        except Exception:
            return False

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str = "smiles",
        add_features: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize an entire DataFrame of molecular data.

        Args:
            df: Input DataFrame with molecular data
            smiles_column: Name of column containing SMILES
            add_features: Whether to add molecular descriptors

        Returns:
            Normalized DataFrame
        """
        if smiles_column not in df.columns:
            logger.error(f"Column '{smiles_column}' not found in DataFrame")
            return df

        normalized_data = []

        for idx, row in df.iterrows():
            smiles = row[smiles_column]

            if pd.isna(smiles):
                continue

            # Canonicalize
            canonical_smiles = self.canonicalize_smiles(smiles)
            if canonical_smiles is None:
                continue

            # Check validity
            if not self.is_valid_molecule(canonical_smiles):
                continue

            # Deduplicate
            if self.remove_duplicates:
                inchikey = self.compute_inchikey(canonical_smiles)
                if inchikey is None:
                    continue
                if inchikey in self._seen_inchikeys:
                    continue
                self._seen_inchikeys.add(inchikey)

            # Create normalized row
            new_row = row.copy()
            new_row[smiles_column] = canonical_smiles

            # Add molecular features if requested
            if add_features:
                mol = Chem.MolFromSmiles(canonical_smiles)
                if mol:
                    new_row["mol_weight"] = Descriptors.MolWt(mol)
                    new_row["logp"] = Descriptors.MolLogP(mol)
                    new_row["num_h_donors"] = Descriptors.NumHDonors(mol)
                    new_row["num_h_acceptors"] = Descriptors.NumHAcceptors(mol)
                    new_row["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
                    new_row["tpsa"] = Descriptors.TPSA(mol)
                    new_row["num_aromatic_rings"] = Descriptors.NumAromaticRings(mol)

            normalized_data.append(new_row)

        result_df = pd.DataFrame(normalized_data)
        logger.info(
            f"Normalized {len(df)} -> {len(result_df)} molecules "
            f"(removed {len(df) - len(result_df)})"
        )

        return result_df

    def merge_datasets(
        self,
        datasets: List[pd.DataFrame],
        smiles_column: str = "smiles",
    ) -> pd.DataFrame:
        """
        Merge multiple datasets with deduplication.

        Args:
            datasets: List of DataFrames to merge
            smiles_column: Name of SMILES column

        Returns:
            Merged and deduplicated DataFrame
        """
        # Concatenate all datasets
        combined = pd.concat(datasets, ignore_index=True)

        # Normalize the combined dataset
        self._seen_inchikeys.clear()  # Reset for fresh deduplication
        normalized = self.normalize_dataframe(combined, smiles_column=smiles_column)

        logger.info(f"Merged {len(datasets)} datasets into {len(normalized)} unique molecules")
        return normalized

    def apply_filters(
        self,
        df: pd.DataFrame,
        smiles_column: str = "smiles",
        lipinski_filter: bool = True,
        molecular_weight_range: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Apply drug-likeness filters to dataset.

        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            lipinski_filter: Apply Lipinski's Rule of Five
            molecular_weight_range: (min, max) molecular weight

        Returns:
            Filtered DataFrame
        """
        filtered = df.copy()

        if lipinski_filter:
            # Lipinski's Rule of Five
            filtered = filtered[
                (filtered["mol_weight"] <= 500)
                & (filtered["logp"] <= 5)
                & (filtered["num_h_donors"] <= 5)
                & (filtered["num_h_acceptors"] <= 10)
            ]

        if molecular_weight_range:
            min_mw, max_mw = molecular_weight_range
            filtered = filtered[
                (filtered["mol_weight"] >= min_mw) & (filtered["mol_weight"] <= max_mw)
            ]

        logger.info(
            f"Applied filters: {len(df)} -> {len(filtered)} molecules "
            f"(removed {len(df) - len(filtered)})"
        )

        return filtered
