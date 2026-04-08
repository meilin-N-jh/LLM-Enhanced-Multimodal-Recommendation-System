"""Distributed Data Parallel (DDP) Trainer for multi-GPU training."""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device, ensure_dir
from src.preprocess import (
    load_data, preprocess_interactions, create_user_item_mapping,
    split_data, create_negative_samples
)
from src.graph_builder import build_user_item_graph, build_item_item_graph, normalize_adjacency, sparse_to_tensor
from src.data_loader import BPRDataset, collate_fn_bpr
from src.models import HybridModel
from src.metrics import compute_metrics


def setup(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, config):
    """Train with DDP."""
    setup(rank, world_size)

    # Set seed
    set_seed(config['training']['seed'] + rank)

    # Get device
    device = torch.device(f"cuda:{rank}")
    print(f"GPU {rank}: Using device: {device}")

    # Load data
    data_dir = Path(config['data']['data_dir'])
    interactions, items, relations = load_data(data_dir)
    print(f"GPU {rank}: Loaded {len(interactions)} interactions")

    # Preprocess
    interactions = preprocess_interactions(interactions)
    user2idx, idx2user, item2idx, idx2item = create_user_item_mapping(interactions)
    n_users = len(user2idx)
    n_items = len(item2idx)

    # Split data
    train_df, val_df, test_df = split_data(
        interactions,
        strategy=config['split']['strategy'],
        test_ratio=config['split']['test_ratio'],
        val_ratio=config['split']['val_ratio']
    )

    if rank == 0:
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load multimodal features
    features_dir = Path(config['data']['output_dir']) / "features"
    fused_embeddings = None

    if features_dir.exists():
        fused_path = features_dir / "fused_embeddings.npy"
        if fused_path.exists():
            fused_embeddings = np.load(fused_path)
            if rank == 0:
                print(f"GPU {rank}: Loaded fused embeddings: {fused_embeddings.shape}")

    # Build graphs
    adj_matrix = build_user_item_graph(train_df, user2idx, item2idx, n_users, n_items)
    adj_matrix = normalize_adjacency(adj_matrix)
    adj_tensor = sparse_to_tensor(adj_matrix).to(device)

    # Build relation graphs
    relation_matrices = {}
    if not config['ablation']['no_relation'] and len(relations) > 0:
        relation_matrices = build_item_item_graph(relations, item2idx, n_items)
        if rank == 0:
            print(f"GPU {rank}: Built {len(relation_matrices)} relation graphs")

    # Create model
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
    )

    # Wrap with DDP
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create training samples
    item_ids = list(item2idx.keys())
    samples = create_negative_samples(train_df, item_ids, n_neg=1, seed=config['training']['seed'])

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        samples,
        num_replicas=world_size,
        rank=rank
    )

    # DataLoader
    dataset = BPRDataset(samples, user2idx, item2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        collate_fn=collate_fn_bpr,
        pin_memory=True
    )

    # Training loop
    epochs = config['training']['epochs']
    best_metric = 0
    best_state = None

    for epoch in range(epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)

        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = dataloader

        for batch in pbar:
            user_idx = batch['user_idx'].to(device)
            pos_idx = batch['pos_item_idx'].to(device)
            neg_idx = batch['neg_item_idx'].to(device)

            optimizer.zero_grad()
            loss = model.module.bpr_loss(user_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Evaluate periodically
            if (epoch + 1) % config['training']['eval_interval'] == 0:
                metrics = evaluate(model, test_df, train_df, user2idx, item2idx, idx2item, config, device)
                print(f"Test metrics: {metrics}")

                hr5 = metrics.get('hr@5', 0)
                if hr5 > best_metric:
                    best_metric = hr5
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save model (only rank 0)
    if rank == 0:
        checkpoint_dir = Path(config['data']['output_dir']) / "checkpoints"
        ensure_dir(checkpoint_dir)

        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint_path = checkpoint_dir / "best_model_ddp.pt"
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'user2idx': user2idx,
            'idx2user': idx2user,
            'item2idx': item2idx,
            'idx2item': idx2item,
            'config': config,
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    cleanup()


def evaluate(model, test_df, train_df, user2idx, item2idx, idx2item, config, device):
    """Evaluate model."""
    model.eval()

    test_users = test_df['user_id'].unique()
    exclude_items = {}

    if train_df is not None:
        for user_id, group in train_df.groupby('user_id'):
            exclude_items[user_id] = set(group['item_id'].tolist())

    predictions = []
    ground_truth = []
    top_n = config['evaluation']['top_n']
    k_values = config['evaluation']['k_values']

    with torch.no_grad():
        for user_id in test_users:
            user_gt = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
            user_idx = user2idx[user_id]
            scores = model.module.predict(user_idx, exclude_items.get(user_id, set()))

            top_items_idx = np.argsort(-scores)[:top_n]
            top_items = [idx2item[i] for i in top_items_idx]

            predictions.append(top_items)
            ground_truth.append(user_gt)

    metrics = compute_metrics(predictions, ground_truth, k_values)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--no_relation", action="store_true")
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override ablation flags
    if args.no_image:
        config['ablation']['no_image'] = True
    if args.no_text:
        config['ablation']['no_text'] = True
    if args.no_relation:
        config['ablation']['no_relation'] = True
    if args.no_rerank:
        config['ablation']['no_rerank'] = True

    # Get number of GPUs
    world_size = args.gpus
    if torch.cuda.is_available():
        world_size = min(world_size, torch.cuda.device_count())
    print(f"Using {world_size} GPUs")

    # Spawn distributed training
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
