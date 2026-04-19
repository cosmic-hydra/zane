"""
Transformer-based Models for Molecular Property Prediction
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularTransformer(nn.Module):
    """
    Transformer model for molecular fingerprints or descriptors
    """

    def __init__(
        self,
        input_dim: int = 2048,  # Fingerprint size
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_seq_len: int = 512,
        model_dim: int | None = None,
        **_: object,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_dim: Output dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = model_dim if model_dim is not None else hidden_dim

        # Input embedding
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_dim, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Predictions
        """
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Project to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Use first token (CLS-like) or mean pooling
        x = x.mean(dim=1)  # [batch_size, hidden_dim]

        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:, : x.size(1), :]
        return self.dropout(x)


class SMILESTransformer(nn.Module):
    """
    Transformer for SMILES strings (character-level)
    """

    def __init__(
        self,
        vocab_size: int = 128,  # ASCII characters
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: torch.Tensor | None = None):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len] (character indices)
            mask: Attention mask

        Returns:
            Predictions
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=mask)

        # Global average pooling
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
