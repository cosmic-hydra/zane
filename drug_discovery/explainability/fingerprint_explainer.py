import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class FingerprintExplainer:
    """
    Explainer for fingerprint-based models.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def attribute_bits(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based bit attribution for fingerprints.
        """
        fingerprint = fingerprint.to(self.device).float()
        fingerprint.requires_grad = True
        
        output = self.model(fingerprint)
        score = output.sum()
        score.backward()
        
        attribution = fingerprint.grad
        return attribution

    def explain_prediction(self, fingerprint: np.ndarray) -> Dict[str, Any]:
        """
        Explain a prediction by identifying key bits in the fingerprint.
        """
        fp_tensor = torch.from_numpy(fingerprint)
        attributions = self.attribute_bits(fp_tensor)
        
        # Identify most influential bits
        attr_np = attributions.detach().cpu().numpy().flatten()
        top_bits = np.argsort(np.abs(attr_np))[::-1][:10]
        
        return {
            "bit_attributions": attr_np.tolist(),
            "top_influential_bits": top_bits.tolist(),
            "explanation_method": "gradient"
        }
