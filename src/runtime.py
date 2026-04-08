"""Shared runtime utilities for training, evaluation, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.graph_builder import (
    build_item_item_graph,
    build_user_item_graph,
    normalize_adjacency,
    sparse_to_tensor,
)
from src.models import HybridModel
from src.preprocess import (
    create_user_item_mapping,
    load_data,
    preprocess_interactions,
    split_data,
)
from src.rerankers import LLMProfileReranker, RuleBasedReranker
from src.utils import get_device


def _clean_text(value: object) -> str:
    """Convert values to a display-safe string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def load_processed_bundle(config: dict) -> dict:
    """Load processed data, splits, and mappings."""
    data_dir = Path(config["data"]["data_dir"])
    interactions, items, relations = load_data(data_dir)

    valid_items = set(items["item_id"].unique())
    interactions = interactions[interactions["item_id"].isin(valid_items)].copy()
    interactions = preprocess_interactions(interactions)

    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"

    if train_file.exists() and val_file.exists() and test_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
    else:
        train_df, val_df, test_df = split_data(
            interactions,
            strategy=config["split"]["strategy"],
            test_ratio=config["split"]["test_ratio"],
            val_ratio=config["split"]["val_ratio"],
        )

    item_ids = set(items["item_id"].unique())
    train_df = train_df[train_df["item_id"].isin(item_ids)].copy()
    val_df = val_df[val_df["item_id"].isin(item_ids)].copy()
    test_df = test_df[test_df["item_id"].isin(item_ids)].copy()

    all_interactions = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(all_interactions)

    return {
        "data_dir": data_dir,
        "interactions": interactions,
        "items": items,
        "relations": relations,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }


def load_aligned_feature_matrix(
    features_dir: Path,
    feature_name: str,
    item2idx: Dict[str, int],
) -> Optional[np.ndarray]:
    """Load a feature matrix and align it to the current item ordering."""
    feature_path = features_dir / f"{feature_name}.npy"
    item_ids_path = features_dir / "item_ids.npy"

    if not feature_path.exists() or not item_ids_path.exists():
        return None

    features = np.load(feature_path)
    feature_item_ids = np.load(item_ids_path, allow_pickle=True)
    id_to_feature_idx = {item_id: idx for idx, item_id in enumerate(feature_item_ids)}

    if features.ndim == 1:
        aligned = np.zeros((len(item2idx),), dtype=np.float32)
    else:
        aligned = np.zeros((len(item2idx), features.shape[1]), dtype=np.float32)
    for item_id, item_idx in item2idx.items():
        feature_idx = id_to_feature_idx.get(item_id)
        if feature_idx is not None:
            aligned[item_idx] = features[feature_idx]

    return aligned


def load_feature_bundle(config: dict, item2idx: Dict[str, int]) -> dict:
    """Load aligned text/image/fused features."""
    features_dir = Path(config["data"]["output_dir"]) / "features"
    return {
        "features_dir": features_dir,
        "text": load_aligned_feature_matrix(features_dir, "text_embeddings", item2idx),
        "image": load_aligned_feature_matrix(features_dir, "image_embeddings", item2idx),
        "image_raw": load_aligned_feature_matrix(features_dir, "image_embeddings_raw", item2idx),
        "image_confidence": load_aligned_feature_matrix(features_dir, "image_confidences", item2idx),
        "fused": load_aligned_feature_matrix(features_dir, "fused_embeddings", item2idx),
    }


def build_user_modality_profiles(
    train_df: pd.DataFrame,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    text_features: Optional[np.ndarray],
    image_features: Optional[np.ndarray],
    image_confidence: Optional[np.ndarray],
    image_confidence_floor: float = 0.05,
    image_confidence_power: float = 1.0,
) -> Dict[str, Optional[np.ndarray]]:
    """Aggregate recency-weighted user text/image preference vectors from training history."""
    user_text_profiles = None
    user_image_profiles = None

    if text_features is not None:
        user_text_profiles = np.zeros((len(user2idx), text_features.shape[1]), dtype=np.float32)
    if image_features is not None:
        user_image_profiles = np.zeros((len(user2idx), image_features.shape[1]), dtype=np.float32)

    if train_df.empty:
        return {"user_text": user_text_profiles, "user_image": user_image_profiles}

    ordered = train_df.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id"):
        user_idx = user2idx.get(user_id)
        if user_idx is None:
            continue

        item_indices = [item2idx[item_id] for item_id in group["item_id"].tolist() if item_id in item2idx]
        if not item_indices:
            continue

        recency_weights = np.linspace(1.0, 2.0, num=len(item_indices), dtype=np.float32)
        if user_text_profiles is not None and text_features is not None:
            text_stack = text_features[item_indices]
            text_weight_sum = float(recency_weights.sum())
            if text_weight_sum > 0:
                user_text_profiles[user_idx] = (text_stack * recency_weights[:, None]).sum(axis=0) / text_weight_sum

        if user_image_profiles is not None and image_features is not None:
            image_stack = image_features[item_indices]
            image_weights = recency_weights.copy()
            if image_confidence is not None:
                conf = np.clip(image_confidence[item_indices], image_confidence_floor, 1.0)
                if image_confidence_power != 1.0:
                    conf = np.power(conf, image_confidence_power)
                image_weights = image_weights * conf
            image_weight_sum = float(image_weights.sum())
            if image_weight_sum > 0:
                user_image_profiles[user_idx] = (image_stack * image_weights[:, None]).sum(axis=0) / image_weight_sum

    return {"user_text": user_text_profiles, "user_image": user_image_profiles}


def build_relation_lookup(relations_df: pd.DataFrame) -> Dict[str, set]:
    """Build item -> related items lookup for reranking."""
    lookup: Dict[str, set] = {}
    for _, row in relations_df.iterrows():
        src = row["item_id"]
        dst = row["related_item_id"]
        lookup.setdefault(src, set()).add(dst)
        lookup.setdefault(dst, set()).add(src)
    return lookup


def build_item_lookup(items_df: pd.DataFrame) -> Dict[str, dict]:
    """Build item metadata lookup."""
    lookup = {}
    for _, row in items_df.iterrows():
        item_id = row["item_id"]
        lookup[item_id] = {
            "title": _clean_text(row.get("title", "")),
            "description": _clean_text(row.get("description", "")),
            "category": _clean_text(row.get("category", "")),
            "brand": _clean_text(row.get("brand", "")),
            "price": _clean_text(row.get("price", "")),
            "image_url": _clean_text(row.get("image_url", "")),
        }
    return lookup


def build_user_history_lookup(interactions_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build timestamp-sorted user history lookup."""
    histories: Dict[str, List[str]] = {}
    if interactions_df.empty:
        return histories

    ordered = interactions_df.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id"):
        histories[user_id] = group["item_id"].tolist()
    return histories


def build_user_feedback_lookup(interactions_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """Build per-user feedback snippets for LLM profile generation."""
    feedback: Dict[str, List[dict]] = {}
    if interactions_df.empty:
        return feedback

    ordered = interactions_df.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id"):
        records = []
        for _, row in group.iterrows():
            records.append(
                {
                    "item_id": row["item_id"],
                    "rating": row.get("rating"),
                    "review_title": _clean_text(row.get("review_title", "")),
                    "review_text": _clean_text(row.get("review_text", "")),
                }
            )
        feedback[user_id] = records
    return feedback


def build_hybrid_model(config: dict, bundle: dict, device: torch.device) -> HybridModel:
    """Construct the hybrid recommender model."""
    user2idx = bundle["user2idx"]
    item2idx = bundle["item2idx"]
    train_df = bundle["train_df"]
    relations = bundle["relations"]
    features = bundle["features"]

    n_users = len(user2idx)
    n_items = len(item2idx)

    adj_matrix = build_user_item_graph(train_df, user2idx, item2idx, n_users, n_items)
    adj_matrix = normalize_adjacency(adj_matrix)
    adj_tensor = sparse_to_tensor(adj_matrix).to(device)

    relation_matrices = {}
    if not config["ablation"]["no_relation"] and len(relations) > 0:
        relation_matrices = build_item_item_graph(relations, item2idx, n_items)

    model = HybridModel(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config["model"]["embedding_dim"],
        adj_tensor=adj_tensor,
        lightgcn_layers=config["model"]["lightgcn_layers"],
        include_ego_embeddings=config["model"].get("include_ego_embeddings", False),
        learnable_layer_weights=config["model"].get("learnable_layer_weights", False),
        text_gate_init=config["model"].get("text_gate_init", 0.0),
        image_gate_init=config["model"].get("image_gate_init", -1.5),
        user_text_gate_init=config["model"].get("user_text_gate_init", -0.4),
        user_image_gate_init=config["model"].get("user_image_gate_init", -1.0),
        image_confidence_floor=config["model"].get("image_confidence_floor", 0.05),
        image_confidence_power=config["model"].get("image_confidence_power", 1.0),
        text_item_scale=config["model"].get("text_item_scale", 0.20),
        image_item_scale=config["model"].get("image_item_scale", 0.15),
        fused_item_scale=config["model"].get("fused_item_scale", 0.15),
        relation_scale=config["model"].get("relation_scale", 0.20),
        align_loss_weight=config["model"].get("align_loss_weight", 0.05),
        bridge_loss_weight=config["model"].get("bridge_loss_weight", 0.02),
        user_text_scale=config["model"].get("user_text_scale", 0.00),
        user_image_scale=config["model"].get("user_image_scale", 0.00),
        text_emb=None if config["ablation"]["no_text"] else features.get("text"),
        image_emb=None if config["ablation"]["no_image"] else features.get("image"),
        image_confidence=None if config["ablation"]["no_image"] else features.get("image_confidence"),
        user_text_profile=None
        if config["ablation"]["no_text"]
        else bundle.get("user_profiles", {}).get("user_text"),
        user_image_profile=None
        if config["ablation"]["no_image"]
        else bundle.get("user_profiles", {}).get("user_image"),
        multimodal_emb=features.get("fused"),
        relation_matrices=relation_matrices,
        use_image=not config["ablation"]["no_image"],
        use_text=not config["ablation"]["no_text"],
        use_relation=not config["ablation"]["no_relation"],
        device=device,
        item2idx=item2idx,
    )
    return model.to(device)


def create_reranker(config: dict, bundle: dict, device: torch.device):
    """Create the configured reranker."""
    if config["ablation"].get("no_rerank", False):
        return None

    rerank_cfg = config.get("reranking", {})
    mode = rerank_cfg.get("mode", "profile")

    text_embeddings = bundle["features"].get("text")
    if text_embeddings is None:
        return None

    item2idx = bundle["item2idx"]
    text_lookup = {item_id: text_embeddings[idx] for item_id, idx in item2idx.items()}
    image_embeddings = None
    if not config["ablation"].get("no_image", False):
        image_matrix = bundle["features"].get("image_raw")
        if image_matrix is not None:
            image_embeddings = {item_id: image_matrix[idx] for item_id, idx in item2idx.items()}
    relation_lookup = build_relation_lookup(bundle["relations"])
    item_lookup = build_item_lookup(bundle["items"])
    user_feedback = build_user_feedback_lookup(bundle["train_df"])

    if mode == "rule":
        return RuleBasedReranker(
            config=config,
            text_embeddings=text_lookup,
            item_relations=relation_lookup,
        )

    return LLMProfileReranker(
        config=config,
        text_embeddings=text_lookup,
        image_embeddings=image_embeddings,
        item_relations=relation_lookup,
        item_metadata=item_lookup,
        user_feedback_lookup=user_feedback,
        device=str(device),
    )


def prepare_runtime_bundle(config: dict, device: Optional[torch.device] = None) -> dict:
    """Load everything needed to train or evaluate the recommender."""
    bundle = load_processed_bundle(config)
    bundle["features"] = load_feature_bundle(config, bundle["item2idx"])
    bundle["user_profiles"] = build_user_modality_profiles(
        bundle["train_df"],
        bundle["user2idx"],
        bundle["item2idx"],
        bundle["features"].get("text"),
        bundle["features"].get("image"),
        bundle["features"].get("image_confidence"),
        image_confidence_floor=config["model"].get("image_confidence_floor", 0.05),
        image_confidence_power=config["model"].get("image_confidence_power", 1.0),
    )
    bundle["device"] = device or get_device(config.get("device", "cuda"))
    return bundle


def load_checkpoint_bundle(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[dict, dict, HybridModel]:
    """Recreate a trained model and its runtime bundle from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config or checkpoint["config"]
    device = device or get_device(config.get("device", "cuda"))

    bundle = prepare_runtime_bundle(config, device=device)

    # Reuse the exact mapping saved during training.
    bundle["user2idx"] = checkpoint["user2idx"]
    bundle["idx2user"] = checkpoint["idx2user"]
    bundle["item2idx"] = checkpoint["item2idx"]
    bundle["idx2item"] = checkpoint["idx2item"]
    valid_users = set(bundle["user2idx"].keys())
    valid_items = set(bundle["item2idx"].keys())
    for split_name in ["train_df", "val_df", "test_df"]:
        frame = bundle[split_name]
        bundle[split_name] = frame[
            frame["user_id"].isin(valid_users) & frame["item_id"].isin(valid_items)
        ].copy()
    bundle["features"] = load_feature_bundle(config, bundle["item2idx"])
    bundle["user_profiles"] = build_user_modality_profiles(
        bundle["train_df"],
        bundle["user2idx"],
        bundle["item2idx"],
        bundle["features"].get("text"),
        bundle["features"].get("image"),
        bundle["features"].get("image_confidence"),
        image_confidence_floor=config["model"].get("image_confidence_floor", 0.05),
        image_confidence_power=config["model"].get("image_confidence_power", 1.0),
    )

    model = build_hybrid_model(config, bundle, device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing_keys:
        print(f"Checkpoint missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Checkpoint unexpected keys: {unexpected_keys}")
    model = model.to(device)
    model.eval()
    return checkpoint, bundle, model


def save_json(data: dict, path: Path) -> None:
    """Save JSON with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
