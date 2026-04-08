"""Multimodal fusion module."""

import numpy as np
import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Fusion module for combining image and text embeddings."""

    def __init__(self, text_dim, image_dim, hidden_dim, output_dim, device="cpu"):
        """Initialize fusion module.

        Args:
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings
            hidden_dim: Hidden dimension for projection
            output_dim: Output dimension
            device: Device to run on
        """
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # Projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_emb, image_emb):
        """Forward pass.

        Args:
            text_emb: Text embeddings (n_items, text_dim)
            image_emb: Image embeddings (n_items, image_dim)

        Returns:
            Fused embeddings (n_items, output_dim)
        """
        # Convert to tensors if needed
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb).float()
        if isinstance(image_emb, np.ndarray):
            image_emb = torch.from_numpy(image_emb).float()

        # Project each modality
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)

        # Fuse by concatenation and sum (simple late fusion)
        # Alternative: could use attention or gating
        fused = text_proj + image_proj

        return fused

    def project_text(self, text_emb):
        """Project text embeddings only."""
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb).float()
        return self.text_proj(text_emb)

    def project_image(self, image_emb):
        """Project image embeddings only."""
        if isinstance(image_emb, np.ndarray):
            image_emb = torch.from_numpy(image_emb).float()
        return self.image_proj(image_emb)
