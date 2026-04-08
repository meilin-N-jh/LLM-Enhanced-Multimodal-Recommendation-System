"""Image encoder using CLIP."""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from io import BytesIO
import urllib.request


class ImageEncoder:
    """Image encoder using CLIP."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        """Initialize image encoder.

        Args:
            model_name: Name of CLIP model
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.output_dim = getattr(self.model.config, "projection_dim", 512)

    def _load_image(self, image_source):
        """Load a PIL image from local path or URL-backed dict."""
        image_path = None
        image_url = None

        if isinstance(image_source, dict):
            image_path = image_source.get("image_path")
            image_url = image_source.get("image_url")
        else:
            image_path = image_source

        if image_path and Path(image_path).exists():
            return Image.open(image_path).convert('RGB')

        if image_url:
            request = urllib.request.Request(
                image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                return Image.open(BytesIO(response.read())).convert('RGB')

        raise FileNotFoundError(f"Image not available: path={image_path}, url={image_url}")

    def encode(self, image_source):
        """Encode single image.

        Args:
            image_source: Local path or dict with image_path/image_url

        Returns:
            numpy array of embedding
        """
        try:
            image = self._load_image(image_source)

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Handle different output formats - get pooled output if available
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state[:, 0, :]  # Take CLS token

            return image_features.cpu().numpy()[0]

        except Exception as e:
            # Return zero vector on error
            print(f"Warning: Failed to encode image {image_source}: {e}")
            return np.zeros(self.output_dim)

    def encode_batch(self, image_sources, batch_size=32):
        """Encode batch of images.

        Args:
            image_sources: List of local paths or dicts with image_path/image_url
            batch_size: Batch size

        Returns:
            numpy array of embeddings
        """
        embeddings = []

        for i in range(0, len(image_sources), batch_size):
            batch_sources = image_sources[i:i + batch_size]
            batch_images = []

            for source in batch_sources:
                try:
                    img = self._load_image(source)
                    batch_images.append(img)
                except Exception:
                    # Use zero image for failed loads
                    batch_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

            # Process batch
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Handle different output formats - get pooled output if available
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state[:, 0, :]  # Take CLS token

            embeddings.append(image_features.cpu().numpy())

        return np.vstack(embeddings)
