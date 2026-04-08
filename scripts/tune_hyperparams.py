#!/usr/bin/env python3
"""
Hyperparameter tuning for hybrid model.
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
import itertools

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed
from src.preprocess import load_data, create_user_item_mapping
from src.graph_builder import build_user_item_graph, normalize_adjacency, sparse_to_tensor, build_item_item_graph
from src.metrics import compute_metrics
from src.models.hybrid_model import HybridModel
from src.data_loader import BPRDataset, collate_fn_bpr


def load_splits(data_dir):
    """Load train/val/test splits."""
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, val_df, test_df


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


def train_and_evaluate(train_df, val_df, test_df, user2idx, item2idx, idx2item,
                      config, device, params):
    """Train model with given params and evaluate."""
    n_users = len(user2idx)
    n_items = len(item2idx)

    # Build graph
    adj_matrix = build_user_item_graph(train_df, user2idx, item2idx, n_users, n_items)
    adj_matrix = normalize_adjacency(adj_matrix)
    adj_tensor = sparse_to_tensor(adj_matrix).to(device)

    # Build relations from train only
    relations_path = config['data']['data_dir'] + "/item_relations.csv"
    relations = pd.read_csv(relations_path) if os.path.exists(relations_path) else pd.DataFrame()
    train_items = set(train_df['item_id'].unique())
    if len(relations) > 0:
        relations = relations[
            (relations['item_id'].isin(train_items)) &
            (relations['related_item_id'].isin(train_items))
        ]
        relation_matrices = build_item_item_graph(relations, item2idx, n_items)
    else:
        relation_matrices = {}

    # Load multimodal features
    fused_embeddings = load_fused_embeddings(config, item2idx, n_items)

    # Create model
    model = HybridModel(
        n_users=n_users,
        n_items=n_items,
        embed_dim=params['embed_dim'],
        adj_tensor=adj_tensor,
        lightgcn_layers=params['n_layers'],
        multimodal_emb=fused_embeddings,
        relation_matrices=relation_matrices,
        use_image=params.get('use_image', True),
        use_text=params.get('use_text', True),
        use_relation=params.get('use_relation', True),
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
        batch_size=params.get('batch_size', 512),
        shuffle=True,
        collate_fn=collate_fn_bpr
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    # Training
    best_val_metric = 0
    best_state = None
    patience_counter = 0

    for epoch in range(params.get('epochs', 100)):
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

        # Evaluate on validation
        if (epoch + 1) % params.get('eval_interval', 5) == 0:
            val_metrics = evaluate_model(model, val_df, train_df, user2idx, item2idx, idx2item, config, device)
            hr10 = val_metrics.get('hr@10', 0)

            if hr10 > best_val_metric:
                best_val_metric = hr10
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= params.get('patience', 10):
                break

    # Load best model and evaluate on test
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_df, train_df, user2idx, item2idx, idx2item, config, device)
    return test_metrics, best_val_metric


def evaluate_model(model, test_df, train_df, user2idx, item2idx, idx2item, config, device):
    """Evaluate model."""
    model.eval()
    test_users = test_df['user_id'].unique()

    exclude_items = {}
    for user_id, group in train_df.groupby('user_id'):
        exclude_items[user_id] = set(group['item_id'].tolist())

    predictions = []
    ground_truth = []
    top_n = config['evaluation']['top_n']

    with torch.no_grad():
        for user_id in test_users:
            user_gt = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
            user_idx = user2idx[user_id]
            scores = model.predict(user_idx, set())

            # Exclude train items
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


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--trials", type=int, default=20, help="Max trials")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(config['data']['data_dir'])
    train_df, val_df, test_df = load_splits(data_dir)
    items = pd.read_csv(data_dir / "items.csv")

    valid_items = set(items['item_id'].unique())
    train_df = train_df[train_df['item_id'].isin(valid_items)]
    val_df = val_df[val_df['item_id'].isin(valid_items)]
    test_df = test_df[test_df['item_id'].isin(valid_items)]

    all_interactions = pd.concat([train_df, val_df, test_df])
    user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(all_interactions)
    n_users = len(user2idx)
    n_items = len(item2idx)

    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define search space
    param_grid = {
        'embed_dim': [32, 64, 128],
        'n_layers': [1, 2, 3],
        'lr': [1e-3, 5e-4, 1e-4],
        'weight_decay': [1e-5, 1e-4],
        'epochs': [100],
        'batch_size': [512],
        'patience': [10],
        'eval_interval': [5],
    }

    # Random search (simplified)
    np.random.seed(42)
    all_configs = list(itertools.product(
        param_grid['embed_dim'],
        param_grid['n_layers'],
        param_grid['lr'],
        param_grid['weight_decay'],
    ))

    # Sample trials
    n_trials = min(args.trials, len(all_configs))
    sampled_configs = [all_configs[i] for i in np.random.choice(len(all_configs), n_trials, replace=False)]

    results = []

    print(f"\nRunning {n_trials} trials...")

    for i, (embed_dim, n_layers, lr, weight_decay) in enumerate(sampled_configs):
        params = {
            'embed_dim': embed_dim,
            'n_layers': n_layers,
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': 100,
            'batch_size': 512,
            'patience': 10,
            'eval_interval': 5,
            'use_image': True,
            'use_text': True,
            'use_relation': True,
        }

        print(f"\nTrial {i+1}/{n_trials}: embed_dim={embed_dim}, n_layers={n_layers}, lr={lr}, weight_decay={weight_decay}")

        try:
            test_metrics, val_metric = train_and_evaluate(
                train_df, val_df, test_df, user2idx, item2idx, idx2item,
                config, device, params
            )

            result = {
                'params': params,
                'val_metric': val_metric,
                'test_hr@5': test_metrics.get('hr@5', 0),
                'test_hr@10': test_metrics.get('hr@10', 0),
                'test_hr@20': test_metrics.get('hr@20', 0),
                'test_ndcg@10': test_metrics.get('ndcg@10', 0),
            }
            results.append(result)

            print(f"  Val HR@10: {val_metric:.4f}, Test HR@10: {result['test_hr@10']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Find best
    if results:
        results_sorted = sorted(results, key=lambda x: x['test_hr@10'], reverse=True)

        print("\n" + "="*60)
        print("TOP 5 RESULTS")
        print("="*60)

        for i, r in enumerate(results_sorted[:5]):
            print(f"\n{i+1}. HR@10 = {r['test_hr@10']:.4f}, NDCG@10 = {r['test_ndcg@10']:.4f}")
            print(f"   embed_dim={r['params']['embed_dim']}, n_layers={r['params']['n_layers']}")
            print(f"   lr={r['params']['lr']}, weight_decay={r['params']['weight_decay']}")

        # Save all results
        output_dir = Path(config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "tuning_results.json", "w") as f:
            json.dump(results_sorted, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir / 'tuning_results.json'}")


if __name__ == "__main__":
    main()
