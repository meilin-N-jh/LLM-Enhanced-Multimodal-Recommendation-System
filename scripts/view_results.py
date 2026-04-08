#!/usr/bin/env python3
"""View training results and make recommendations."""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.preprocess import load_data, preprocess_interactions, create_user_item_mapping
from src.metrics import compute_metrics


def view_training_results(checkpoint_path="outputs/checkpoints/best_model.pt"):
    """View saved training results."""
    print("=" * 60)
    print("训练结果查看")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n模型配置:")
    config = checkpoint.get('config', {})
    print(f"  - embedding维度: {config.get('model', {}).get('embedding_dim')}")
    print(f"  - LightGCN层数: {config.get('model', {}).get('lightgcn_layers')}")
    print(f"  - 训练轮数: {config.get('training', {}).get('epochs')}")
    print(f"  - 批量大小: {config.get('training', {}).get('batch_size')}")

    print(f"\n数据集:")
    user2idx = checkpoint['user2idx']
    item2idx = checkpoint['item2idx']
    print(f"  - 用户数: {len(user2idx)}")
    print(f"  - 商品数: {len(item2idx)}")

    print(f"\n评估指标:")
    metrics = checkpoint.get('metrics', {})
    if metrics:
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                v = v.item()
            print(f"  - {k}: {v:.4f}")
    else:
        print("  - 无保存的指标")

    return checkpoint


def recommend_items(checkpoint_path, user_id, top_n=10):
    """Get recommendations for a user."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']
    user2idx = checkpoint['user2idx']
    idx2user = checkpoint['idx2user']
    item2idx = checkpoint['item2idx']
    idx2item = checkpoint['idx2item']

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data for model setup
    from src.graph_builder import build_user_item_graph, normalize_adjacency, sparse_to_tensor
    from src.models import HybridModel

    data_dir = Path(config['data']['data_dir'])
    interactions, items, relations = load_data(data_dir)
    interactions = preprocess_interactions(interactions)

    # Build graph
    adj_matrix = build_user_item_graph(interactions, user2idx, item2idx, len(user2idx), len(item2idx))
    adj_matrix = normalize_adjacency(adj_matrix)
    adj_tensor = sparse_to_tensor(adj_matrix).to(device)

    # Load multimodal features
    features_dir = Path(config['data']['output_dir']) / "features"
    fused_embeddings = None
    if features_dir.exists():
        fused_path = features_dir / "fused_embeddings.npy"
        if fused_path.exists():
            fused_embeddings = np.load(fused_path)

    # Build relation graphs
    from src.graph_builder import build_item_item_graph
    relation_matrices = build_item_item_graph(relations, item2idx, len(item2idx))

    # Create model
    model = HybridModel(
        n_users=len(user2idx),
        n_items=len(item2idx),
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
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get recommendations
    if user_id not in user2idx:
        print(f"用户 {user_id} 不存在!")
        return

    user_idx = user2idx[user_id]

    with torch.no_grad():
        scores = model.predict(user_idx, exclude_items=set())

    # Get top N
    top_indices = np.argsort(-scores)[:top_n]

    print(f"\n为用户 {user_id} 的推荐 (Top {top_n}):")
    print("-" * 40)
    for i, idx in enumerate(top_indices):
        item_id = idx2item[idx]
        score = scores[idx]
        print(f"  {i+1}. {item_id} (分数: {score:.4f})")

    return item_id, score


def compare_users(checkpoint_path, user_ids, top_n=5):
    """Compare recommendations for multiple users."""
    print("\n" + "=" * 60)
    print("用户推荐对比")
    print("=" * 60)

    for user_id in user_ids:
        print(f"\n用户 {user_id}:")
        recommend_items(checkpoint_path, user_id, top_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="查看训练结果")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--user", type=str, default="u1")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--compare", nargs='+', help="比较多个用户")
    args = parser.parse_args()

    # View results
    view_training_results(args.checkpoint)

    if args.compare:
        compare_users(args.checkpoint, args.compare, args.top_n)
    else:
        recommend_items(args.checkpoint, args.user, args.top_n)
