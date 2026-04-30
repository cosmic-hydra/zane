import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from typing import Dict, List, Optional, Tuple, Any

class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for multi-relational drug discovery.
    Supports DRUG, PROTEIN, GENE, DISEASE node types and their interactions.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Node encoders for each type
        self.node_encoders = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in node_types
        })
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                # edge_type is (src, rel, dst)
                conv_dict[edge_type] = GATConv((-1, -1), hidden_dim, add_self_loops=False)
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
        # Final prediction heads
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # Encode nodes
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_encoders[node_type](x).relu()
            
        # Multi-layer heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict.keys():
                x_dict[node_type] = x_dict[node_type].relu()
                x_dict[node_type] = self.dropout(x_dict[node_type])
                
        # Return hidden representations (can be used for further tasks)
        return x_dict

    def predict_link(self, h_dict: Dict[str, torch.Tensor], edge_type: Tuple[str, str, str], edge_index: torch.Tensor):
        """
        Predict link presence or property between two nodes.
        """
        src_type, _, dst_type = edge_type
        src_h = h_dict[src_type][edge_index[0]]
        dst_h = h_dict[dst_type][edge_index[1]]
        
        # Combine representations
        h = src_h * dst_h
        
        h = self.fc1(h).relu()
        h = self.dropout(h)
        return self.fc2(h)
