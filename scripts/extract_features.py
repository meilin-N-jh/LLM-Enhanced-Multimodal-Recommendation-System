#!/usr/bin/env python3
"""Extract multimodal features from items."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_encoder import TextEncoder
from src.image_encoder import ImageEncoder
from src.fusion import FusionModule
from src.utils import load_config


def l2_normalize(matrix, eps=1e-12):
    """L2-normalize a 2D numpy matrix row-wise."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def learn_image_to_text_mapping(image_embeddings, text_embeddings, reg_lambda=0.05):
    """Fit a regularized linear map from image space to text space."""
    x = image_embeddings.astype(np.float32)
    y = text_embeddings.astype(np.float32)
    xtx = x.T @ x
    reg = reg_lambda * np.eye(xtx.shape[0], dtype=np.float32)
    w = np.linalg.solve(xtx + reg, x.T @ y)
    return w.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract multimodal features")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (overrides config)")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Only encode the first N items")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set data directory
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["data_dir"])
    output_dir = Path(config["data"]["output_dir"])
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")

    # Load items
    items_df = pd.read_csv(data_dir / "items.csv")
    if args.max_items:
        items_df = items_df.head(args.max_items).copy()
    print(f"Loaded {len(items_df)} items")

    # Get device
    device = config.get("device", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    print(f"Using device: {device}")

    # Extract text features
    print("\n[1/3] Extracting text features...")
    # Use local model if available
    text_model_path = "models/all-MiniLM-L6-v2"
    text_model_name = text_model_path if os.path.exists(text_model_path) else config["model"]["text_encoder_model"]
    text_encoder = TextEncoder(
        model_name=text_model_name,
        device=device
    )

    # Build text from item fields
    texts = []
    for _, row in items_df.iterrows():
        title = "" if pd.isna(row['title']) else str(row['title'])
        description = "" if pd.isna(row['description']) else str(row['description'])
        category = "" if pd.isna(row['category']) else str(row['category'])
        brand = "" if pd.isna(row['brand']) else str(row['brand'])
        text = f"{title}. {description}. Category: {category}. Brand: {brand}."
        texts.append(text)

    text_embeddings = text_encoder.encode(texts, batch_size=32).astype(np.float32)
    text_embeddings = l2_normalize(text_embeddings)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # Save text embeddings
    np.save(features_dir / "text_embeddings.npy", text_embeddings)
    print(f"Saved text embeddings to {features_dir / 'text_embeddings.npy'}")

    # Extract image features
    print("\n[2/3] Extracting image features...")
    # Use local model if available
    clip_model_path = "models/clip-vit-large-patch14"
    clip_model_name = clip_model_path if os.path.exists(clip_model_path) else config["model"]["image_encoder_model"]
    image_encoder = ImageEncoder(
        model_name=clip_model_name,
        device=device
    )

    image_sources = []
    for _, row in items_df.iterrows():
        local_path = ""
        if not pd.isna(row.get("image_path")):
            local_path = str(data_dir / row["image_path"])
        image_url = "" if pd.isna(row.get("image_url")) else str(row["image_url"])
        image_sources.append(
            {
                "image_path": local_path,
                "image_url": image_url,
            }
        )
    image_embeddings_raw = image_encoder.encode_batch(image_sources).astype(np.float32)
    image_embeddings_raw = l2_normalize(image_embeddings_raw)
    print(f"Image embeddings shape: {image_embeddings_raw.shape}")

    print("Learning image-to-text alignment...")
    image_to_text_w = learn_image_to_text_mapping(image_embeddings_raw, text_embeddings)
    image_embeddings = image_embeddings_raw @ image_to_text_w
    image_embeddings = l2_normalize(image_embeddings.astype(np.float32))
    image_confidences = np.sum(image_embeddings * text_embeddings, axis=1)
    image_confidences = np.clip((image_confidences + 1.0) / 2.0, 0.05, 1.0).astype(np.float32)

    # Save image embeddings
    np.save(features_dir / "image_embeddings.npy", image_embeddings)
    print(f"Saved image embeddings to {features_dir / 'image_embeddings.npy'}")
    np.save(features_dir / "image_embeddings_raw.npy", image_embeddings_raw)
    print(f"Saved raw image embeddings to {features_dir / 'image_embeddings_raw.npy'}")
    np.save(features_dir / "image_confidences.npy", image_confidences)
    print(f"Saved image confidence scores to {features_dir / 'image_confidences.npy'}")

    # Create fused features
    print("\n[3/3] Creating fused multimodal features...")
    fusion = FusionModule(
        text_dim=text_embeddings.shape[1],
        image_dim=image_embeddings.shape[1],
        hidden_dim=config["model"]["fusion_hidden_dim"],
        output_dim=config["model"]["embedding_dim"],
        device=device
    )

    fused_embeddings = fusion(text_embeddings, image_embeddings)
    fused_embeddings = fused_embeddings.detach().cpu().numpy().astype(np.float32)
    fused_embeddings = l2_normalize(fused_embeddings)
    print(f"Fused embeddings shape: {fused_embeddings.shape}")

    # Save fused embeddings
    np.save(features_dir / "fused_embeddings.npy", fused_embeddings)
    print(f"Saved fused embeddings to {features_dir / 'fused_embeddings.npy'}")

    # Save item IDs
    item_ids = items_df["item_id"].tolist()
    np.save(features_dir / "item_ids.npy", np.array(item_ids))
    print(f"Saved item IDs to {features_dir / 'item_ids.npy'}")

    print("\nFeature extraction complete!")
    print(f"Outputs saved to: {features_dir}")


if __name__ == "__main__":
    main()
