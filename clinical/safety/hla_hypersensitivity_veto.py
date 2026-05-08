import torch
import torch.nn as nn
import pandas as pd
import json
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LethalHypersensitivityVeto(Exception):
    """Exception raised when a drug is predicted to trigger a lethal HLA-mediated autoimmune reaction."""
    pass

class PersonalizedImmunotoxScreener:
    """
    Evaluates potential for lethal idiosyncratic hypersensitivity reactions (e.g., SJS/TEN)
    by predicting drug binding to patient-specific HLA alleles.
    """
    def __init__(self):
        # Mock geometric deep learning surrogate model for HLA-drug binding
        # In production, this would be a GNN-based binding affinity predictor
        self.binding_model = nn.Sequential(
            nn.Linear(2048 + 512, 1024), # Drug fingerprint + HLA embedding
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.hla_embeddings: Dict[str, torch.Tensor] = {} # Mock embeddings for common alleles

    def ingest_patient_hla_typing(self, hla_json_path: str) -> List[str]:
        """
        Parses the patient's exact HLA class I and II genotypes.
        """
        try:
            with open(hla_json_path, 'r') as f:
                data = json.load(f)
            alleles = data.get("hla_typing", [])
            logger.info(f"Loaded {len(alleles)} patient HLA alleles: {alleles}")
            return alleles
        except Exception as e:
            logger.error(f"Failed to parse HLA typing at {hla_json_path}: {str(e)}")
            return ["HLA-B*57:01"] # High-risk default if unknown

    def predict_hla_drug_complex(self, smiles: str, patient_hla_alleles: List[str]) -> float:
        """
        Predicts if the generated molecule will bind into the patient's HLA antigen-presentation groove.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0

        # Generate Morgan Fingerprint for the drug
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        drug_tensor = torch.tensor(list(fp), dtype=torch.float32)

        max_risk = 0.0
        for allele in patient_hla_alleles:
            # Get or generate mock HLA embedding
            hla_emb = self.hla_embeddings.get(allele, torch.randn(512))
            
            # Predict binding risk
            combined = torch.cat([drug_tensor, hla_emb])
            risk_score = self.binding_model(combined).item()
            
            if risk_score > 0.85: # Threshold for triggering a T-cell response
                logger.error(f"LETHAL VETO: High risk of HLA-mediated hypersensitivity with allele {allele}")
                raise LethalHypersensitivityVeto(
                    f"Molecule predicted to bind into {allele} groove, likely triggering lethal autoimmune reaction."
                )
            
            max_risk = max(max_risk, risk_score)
            
        return max_risk
