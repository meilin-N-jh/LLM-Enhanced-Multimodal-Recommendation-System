#!/usr/bin/env python3
"""
Run all baseline experiments and comparison.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed
from src.preprocess import load_data, preprocess_interactions, create_user_item_mapping
from src.graph_builder import build_user_item_graph, normalize_adjacency, sparse_to_tensor
from src.metrics import compute_metrics
from src.models.popularity import Popularity
from src.models.mf import MF
from src.models.bpr import BPRMF
from src.models.lightgcn import LightGCN
from src.models.hybrid_model import HybridModel
from src.data_loader import BPRDataset, collate_fn_bpr


def load_splits(data_dir):
    """Load train/val/test splits."""
    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"

    if train_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file) if val_file.exists() else None
        test_df = pd.read_csv(test_file)
    else:
        # Fallback to interactions
        interactions = pd.read_csv(data_dir / "interactions.csv")
        from src.preprocess import split_data
        config = load_config("configs/default.yaml")
        train_df, val_df, test_df = split_data(
            interactions,
            strategy=config['split']['strategy'],
            test_ratio=config['split']['test_ratio'],
            val_ratio=config['split']['val_ratio']
        )

    return train_df, val_df, test_df


def evaluate_model(model, test_df, train_df, user2idx, item2idx, idx2item, config, device, is_popularity=False):
    """Evaluate a model."""
    model.eval()

    test_users = test_df['user_id'].unique()
    exclude_items = {}

    if train_df is not None and not is_popularity:
        for user_id, group in train_df.groupby('user_id'):
            exclude_items[user_id] = set(group['item_id'].tolist())

    predictions = []
    ground_truth = []
    top_n = config['evaluation']['top_n']

    with torch.no_grad():
        for user_id in test_users:
            user_gt = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
            user_idx = user2idx[user_id]

            # Get base scores
            if hasattr(model, 'predict'):
                scores = model.predict(user_idx, exclude_items.get(user_id, set()))
            else:
                scores = model.get_scores(user_idx)

            # For non-popularity models, exclude train items
            if not is_popularity:
                for item in exclude_items.get(user_id, []):
                    if item in item2idx:
                        scores[item2idx[item]] = -999999

            top_items_idx = np.argsort(-scores)[:top_n]
            top_items = [list(item2idx.keys())[i] for i in top_items_idx]

            predictions.append(top_items)
            ground_truth.append(user_gt)

    k_values = config['evaluation']['k_values']
    metrics = compute_metrics(predictions, ground_truth, k_values)
    return metrics


def train_popularity(train_df, test_df, user2idx, item2idx, config, device):
    """Train Popularity baseline."""
    print("\n=== Training Popularity ===")
    n_items = len(item2idx)
    model = Popularity(n_items)
    model.fit(train_df, item2idx)
    return model


def train_mf(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor=None):
    """Train MF baseline."""
    print("\n=== Training MF ===")

    n_users = len(user2idx)
    n_items = len(item2idx)

    model = MF(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config['model']['embedding_dim']
    ).to(device)

    # Create training samples
    from src.preprocess import create_negative_samples
    item_ids = list(item2idx.keys())
    samples = create_negative_samples(train_df, item_ids, n_neg=1, seed=config['training']['seed'])

    dataset = BPRDataset(samples, user2idx, item2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_bpr
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(30):  # Quick training
        model.train()
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            pos_idx = batch['pos_item_idx'].to(device)
            neg_idx = batch['neg_item_idx'].to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()

    return model


def train_bpr_mf(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor=None):
    """Train BPR-MF baseline."""
    print("\n=== Training BPR-MF ===")

    n_users = len(user2idx)
    n_items = len(item2idx)

    model = BPRMF(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config['model']['embedding_dim']
    ).to(device)

    # Create training samples
    from src.preprocess import create_negative_samples
    item_ids = list(item2idx.keys())
    samples = create_negative_samples(train_df, item_ids, n_neg=1, seed=config['training']['seed'])

    dataset = BPRDataset(samples, user2idx, item2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_bpr
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            pos_idx = batch['pos_item_idx'].to(device)
            neg_idx = batch['neg_item_idx'].to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model


def train_lightgcn(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor):
    """Train Pure LightGCN (no multimodal)."""
    print("\n=== Training Pure LightGCN ===")

    n_users = len(user2idx)
    n_items = len(item2idx)

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config['model']['embedding_dim'],
        adj_tensor=adj_tensor,
        n_layers=config['model']['lightgcn_layers']
    ).to(device)

    # Create training samples
    from src.preprocess import create_negative_samples
    item_ids = list(item2idx.keys())
    samples = create_negative_samples(train_df, item_ids, n_neg=1, seed=config['training']['seed'])

    dataset = BPRDataset(samples, user2idx, item2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_bpr
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            pos_idx = batch['pos_item_idx'].to(device)
            neg_idx = batch['neg_item_idx'].to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model


def train_hybrid(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor, fused_embeddings):
    """Train Full Hybrid Model."""
    print("\n=== Training Full Hybrid Model ===")

    n_users = len(user2idx)
    n_items = len(item2idx)

    # Build relation graphs from TRAIN ONLY to avoid data leakage
    from src.graph_builder import build_item_item_graph

    # Method 1: Use relations from item_relations.csv but filter to train items only
    relations_path = config['data']['data_dir'] + "/item_relations.csv"
    relations = pd.read_csv(relations_path) if os.path.exists(relations_path) else pd.DataFrame()

    # Filter relations: both item_id and related_item_id should be in training set
    train_items = set(train_df['item_id'].unique())
    if len(relations) > 0:
        relations = relations[
            (relations['item_id'].isin(train_items)) &
            (relations['related_item_id'].isin(train_items))
        ]
        print(f"  Filtered relations to {len(relations)} (train only)")
        relation_matrices = build_item_item_graph(relations, item2idx, n_items)
    else:
        relation_matrices = {}

    model = HybridModel(
        n_users=n_users,
        n_items=n_items,
        embed_dim=config['model']['embedding_dim'],
        adj_tensor=adj_tensor,
        lightgcn_layers=config['model']['lightgcn_layers'],
        multimodal_emb=fused_embeddings,
        relation_matrices=relation_matrices,
        use_image=not config['ablation']['no_image'],
        use_text=not config['ablation']['no_text'],
        use_relation=not config['ablation']['no_relation'],
        device=device,
        item2idx=item2idx
    ).to(device)

    # Create training samples
    from src.preprocess import create_negative_samples
    item_ids = list(item2idx.keys())
    samples = create_negative_samples(train_df, item_ids, n_neg=1, seed=config['training']['seed'])

    dataset = BPRDataset(samples, user2idx, item2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_bpr
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            pos_idx = batch['pos_item_idx'].to(device)
            neg_idx = batch['neg_item_idx'].to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model


def load_fused_embeddings(config, item2idx, n_items):
    """Load and align fused embeddings."""
    features_dir = Path(config['data']['output_dir']) / "features"
    fused_path = features_dir / "fused_embeddings.npy"
    item_ids_path = features_dir / "item_ids.npy"

    if not fused_path.exists():
        return None

    all_fused = np.load(fused_path)
    all_item_ids = np.load(item_ids_path, allow_pickle=True)

    item_id_to_feat_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}
    fused_embeddings = np.zeros((n_items, all_fused.shape[1]))

    for item_id, item_idx in item2idx.items():
        if item_id in item_id_to_feat_idx:
            feat_idx = item_id_to_feat_idx[item_id]
            fused_embeddings[item_idx] = all_fused[feat_idx]

    return fused_embeddings


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=["all"],
                       help="Models to run: popularity, mf, bpr, lightgcn, hybrid, all")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config['training']['seed'])

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(config['data']['data_dir'])
    interactions, items, relations = load_data(data_dir)
    print(f"Loaded {len(interactions)} interactions, {len(items)} items")

    # Filter valid items
    valid_items = set(items['item_id'].unique())
    interactions = interactions[interactions['item_id'].isin(valid_items)]

    # Load splits
    train_df, val_df, test_df = load_splits(data_dir)
    print(f"Splits - Train: {len(train_df)}, Test: {len(test_df)}")

    # Create mappings
    from src.preprocess import preprocess_interactions
    all_interactions = pd.concat([train_df, test_df])
    user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(all_interactions)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"Users: {n_users}, Items: {n_items}")

    # Build graph
    adj_matrix = build_user_item_graph(train_df, user2idx, item2idx, n_users, n_items)
    adj_matrix = normalize_adjacency(adj_matrix)
    adj_tensor = sparse_to_tensor(adj_matrix).to(device)

    # Load fused embeddings
    fused_embeddings = load_fused_embeddings(config, item2idx, n_items)

    # Results storage
    results = {}

    # Determine models to run
    if "all" in args.models:
        models_to_run = ["popularity", "mf", "bpr", "lightgcn", "hybrid"]
    else:
        models_to_run = args.models

    # Run each model
    for model_name in models_to_run:
        print(f"\n{'='*50}")
        print(f"Running: {model_name}")
        print(f"{'='*50}")

        if model_name == "popularity":
            model = train_popularity(train_df, test_df, user2idx, item2idx, config, device)
        elif model_name == "mf":
            model = train_mf(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor)
        elif model_name == "bpr":
            model = train_bpr_mf(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor)
        elif model_name == "lightgcn":
            model = train_lightgcn(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor)
        elif model_name == "hybrid":
            # Train hybrid model
            model = train_hybrid(train_df, test_df, user2idx, item2idx, idx2item, config, device, adj_tensor, fused_embeddings)
        else:
            continue

        # Evaluate
        print(f"  Evaluating {model_name}...")
        is_popularity = (model_name == "popularity")
        metrics = evaluate_model(model, test_df, train_df, user2idx, item2idx, idx2item, config, device, is_popularity)
        results[model_name] = metrics

        print(f"\n  Results for {model_name}:")
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                v = v.item()
            print(f"    {k}: {v:.4f}")

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)

    # Header
    print(f"{'Model':<20}", end="")
    for k in [5, 10, 20]:
        print(f"{'HR@'+str(k):<12}{'NDCG@'+str(k):<12}", end="")
    print()

    # Rows
    for model_name in models_to_run:
        if model_name not in results:
            continue
        print(f"{model_name:<20}", end="")
        for k in [5, 10, 20]:
            hr = results[model_name].get(f'hr@{k}', 0)
            ndcg = results[model_name].get(f'ndcg@{k}', 0)
            if hasattr(hr, 'item'):
                hr = hr.item()
            if hasattr(ndcg, 'item'):
                ndcg = ndcg.item()
            print(f"{hr:<12.4f}{ndcg:<12.4f}", end="")
        print()

    # Save results
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
