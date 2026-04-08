"""Text encoder using sentence-transformers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """Text encoder using sentence-transformers."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        """Initialize text encoder.

        Args:
            model_name: Name of sentence-transformer model
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

    def encode(self, texts, batch_size=32, show_progress=True):
        """Encode texts to embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def encode_single(self, text):
        """Encode single text.

        Args:
            text: Text string

        Returns:
            numpy array of embedding
        """
        return self.model.encode([text], convert_to_numpy=True)[0]
