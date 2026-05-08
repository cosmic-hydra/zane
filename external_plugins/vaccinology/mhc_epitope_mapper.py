import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PatientEpitopeMapper:
    """
    Simulates binding affinity of viral peptides against patient-specific MHC alleles.
    """
    def __init__(self):
        # Mock neural network for binding prediction (similar to NetMHCpan architecture)
        self.model = nn.Sequential(
            nn.Linear(20 * 9, 128), # 9-mer peptides, 20 amino acids
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def _encode_peptide(self, peptide: str) -> torch.Tensor:
        """One-hot encodes a 9-mer peptide."""
        tensor = torch.zeros(9, 20)
        for i, aa in enumerate(peptide):
            if aa in self.aa_map:
                tensor[i, self.aa_map[aa]] = 1.0
        return tensor.flatten()

    def predict_mhc_class_I_binding(self, viral_fasta: str, patient_hla_alleles: List[str]) -> pd.DataFrame:
        """
        Predicts binding affinity (K_d) for all 9-mer peptides in the viral sequence.
        """
        results = []
        try:
            records = list(SeqIO.parse(viral_fasta, "fasta"))
            for record in records:
                sequence = str(record.seq)
                # Sliding window of 9 amino acids
                for i in range(len(sequence) - 8):
                    peptide = sequence[i:i+9]
                    
                    # Simulated prediction logic
                    # In a real system, this would be conditioned on the HLA allele
                    peptide_tensor = self._encode_peptide(peptide)
                    binding_score = self.model(peptide_tensor).item()
                    
                    # Convert score to nanomolar Kd (Log scale mapping)
                    # High score = Low Kd (strong binding)
                    kd_nm = 50000**(1 - binding_score)
                    
                    results.append({
                        "peptide": peptide,
                        "start_pos": i,
                        "kd_nm": kd_nm,
                        "allele": patient_hla_alleles[0] if patient_hla_alleles else "HLA-A*02:01",
                        "immunogenicity_score": binding_score
                    })
                    
            df = pd.DataFrame(results)
            # Filter for "strong binders" (Kd < 50nM)
            return df.sort_values(by="kd_nm").reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Epitope mapping failed: {str(e)}")
            return pd.DataFrame()

    def select_optimal_antigen_payload(self, viral_fasta: str, patient_hla_alleles: List[str]) -> str:
        """
        Constructs a synthetic poly-epitope chain containing highly immunogenic fragments.
        """
        predictions = self.predict_mhc_class_I_binding(viral_fasta, patient_hla_alleles)
        
        if predictions.empty:
            return ""
            
        # Select top 5 non-overlapping epitopes
        top_epitopes = predictions.head(5)["peptide"].tolist()
        
        # Link epitopes with flexible spacers (e.g., AAY linkers)
        poly_epitope = "AAY".join(top_epitopes)
        
        logger.info(f"Generated optimized poly-epitope payload of length {len(poly_epitope)}")
        return poly_epitope
