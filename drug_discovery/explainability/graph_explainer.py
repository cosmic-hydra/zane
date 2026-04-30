import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from torch_geometric.data import Data

class GraphExplainer:
    """
    Explainer for Graph Neural Network models.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def attribute_nodes(self, data: Data, target_class: int = 0) -> torch.Tensor:
        """
        Simple gradient-based node attribution.
        Higher values mean the node is more important for the prediction.
        """
        data = data.to(self.device)
        data.x.requires_grad = True
        
        output = self.model(data)
        
        if output.dim() > 1 and output.size(1) > 1:
            score = output[0, target_class]
        else:
            score = output[0]
            
        score.backward()
        
        # Attribution is the norm of the gradient
        attribution = data.x.grad.norm(dim=1)
        return attribution

    def explain_prediction(self, data: Data) -> Dict[str, Any]:
        """
        Generate a full explanation for a graph prediction.
        """
        attributions = self.attribute_nodes(data)
        
        # Get top-k important nodes
        top_indices = torch.argsort(attributions, descending=True)[:5]
        
        return {
            "node_attributions": attributions.tolist(),
            "top_nodes": top_indices.tolist(),
            "explanation_method": "gradient_norm"
        }
