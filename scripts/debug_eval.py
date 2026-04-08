#!/usr/bin/env python3
"""Debug script to find the evaluation bug."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, '.')

from src.utils import load_config
from src.preprocess import load_data, create_user_item_mapping
from src.graph_builder import build_user_item_graph, normalize_adjacency, sparse_to_tensor
from src.models.hybrid_model import HybridModel

config = load_config('configs/default.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_dir = Path(config['data']['data_dir'])
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
items = pd.read_csv(data_dir / "items.csv")

print("="*60)
print("DATA CHECK")
print("="*60)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Items: {len(items)}")

# Filter valid items
valid_items = set(items['item_id'].unique())
train_df = train_df[train_df['item_id'].isin(valid_items)]
test_df = test_df[test_df['item_id'].isin(valid_items)]
print(f"After filter - Train: {len(train_df)}, Test: {len(test_df)}")

# Create mappings
all_interactions = pd.concat([train_df, test_df])
user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(all_interactions)
n_users = len(user2idx)
n_items = len(item2idx)
print(f"Users: {n_users}, Items: {n_items}")

# Check test items
test_items = set(test_df['item_id'].unique())
train_items = set(train_df['item_id'].unique())
print(f"\nTest items: {len(test_items)}")
print(f"Train items: {len(train_items)}")
print(f"Test items in train: {len(test_items & train_items)}")

# Build graph
adj = build_user_item_graph(train_df, user2idx, item2idx, n_users, n_items)
adj = normalize_adjacency(adj)
adj_tensor = sparse_to_tensor(adj)

# Load fused embeddings
features_dir = Path(config['data']['output_dir']) / "features"
fused = np.load(features_dir / "fused_embeddings.npy")
item_ids = np.load(features_dir / "item_ids.npy", allow_pickle=True)
print(f"\nFused embeddings: {fused.shape}")
print(f"Feature items: {len(item_ids)}")

# Check alignment
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
items_not_in_features = [i for i in item2idx.keys() if i not in item_id_to_idx]
print(f"Items not in features: {len(items_not_in_features)}")

# Load model
print("\n" + "="*60)
print("MODEL CHECK")
print("="*60)

model = HybridModel(
    n_users=n_users,
    n_items=n_items,
    embed_dim=64,
    adj_tensor=adj_tensor,
    lightgcn_layers=3,
    multimodal_emb=fused,
    relation_matrices={},
    use_image=False,
    use_text=False,
    use_relation=False,
    device=device,
    item2idx=item2idx
)

# Train the model briefly
from src.preprocess import create_negative_samples
from src.data_loader import BPRDataset, collate_fn_bpr

item_ids_list = list(item2idx.keys())
samples = create_negative_samples(train_df, item_ids_list, n_neg=1, seed=42)
dataset = BPRDataset(samples, user2idx, item2idx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate_fn_bpr)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for batch in dataloader:
    user_idx = batch['user_idx'].to(device)
    pos_idx = batch['pos_item_idx'].to(device)
    neg_idx = batch['neg_item_idx'].to(device)
    optimizer.zero_grad()
    try:
        loss = model.bpr_loss(user_idx, pos_idx, neg_idx)
        loss.backward()
        optimizer.step()
    except Exception as e:
        print(f"Training error: {e}")
        break

print("Model initialized (using random weights for debugging)")

print("Model trained briefly for debugging")

# Detailed evaluation for first 5 users
print("\n" + "="*60)
print("DETAILED EVALUATION - FIRST 5 USERS")
print("="*60)

model.eval()
test_users = test_df['user_id'].unique()[:5]

for user_id in test_users:
    user_gt = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
    user_train_items = set(train_df[train_df['user_id'] == user_id]['item_id'].tolist())

    print(f"\n--- User: {user_id} ---")
    print(f"Ground truth: {user_gt}")
    print(f"Train items ({len(user_train_items)}): {list(user_train_items)[:5]}...")
    print(f"Test item in train: {user_gt[0] in user_train_items}")

    user_idx = user2idx[user_id]
    with torch.no_grad():
        scores = model.predict(user_idx, set())

    print(f"Score range before exclusion: min={scores.min():.4f}, max={scores.max():.4f}")

    # Exclude train items
    excluded_count = 0
    for item in user_train_items:
        if item in item2idx:
            scores[item2idx[item]] = -999999
            excluded_count += 1
    print(f"Excluded {excluded_count} train items")

    # Get top 10
    top_idx = np.argsort(-scores)[:10]
    top_items = [idx2item[i] for i in top_idx]

    print(f"Top 10: {top_items}")
    print(f"Hit at 5: {any(gt in top_items[:5] for gt in user_gt)}")
    print(f"Hit at 10: {any(gt in top_items for gt in user_gt)}")

    # Check if test item is in top 10
    if user_gt[0] in top_items:
        rank = top_items.index(user_gt[0]) + 1
        print(f">>> HIT! Rank: {rank}")
    else:
        print(f">>> MISS! Score of test item: {scores[item2idx[user_gt[0]]]:.4f}")
