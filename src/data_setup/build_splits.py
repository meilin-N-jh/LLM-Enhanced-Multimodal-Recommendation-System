#!/usr/bin/env python3
"""
Build train/val/test splits for Amazon All_Beauty dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    import yaml
    config_path = Path("configs/data_sources.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def time_based_split(interactions_df, test_ratio=0.2, val_ratio=0.1):
    """
    Split data based on timestamp.
    For each user, use last test_ratio items for test,
    last val_ratio (before test) for validation.
    """
    logger.info("Creating time-based split...")

    # Sort by user and timestamp
    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

    train_data = []
    val_data = []
    test_data = []

    for user_id, group in interactions_df.groupby('user_id'):
        group = group.sort_values('timestamp')
        n = len(group)

        if n < 3:
            # Not enough data, put all in train
            train_data.append(group)
            continue

        # Calculate split indices
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_data.append(group.iloc[-n_test:])
        val_data.append(group.iloc[-(n_test + n_val):-n_test])
        train_data.append(group.iloc[:-(n_test + n_val)])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df


def random_split(interactions_df, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Random split of data.
    """
    logger.info("Creating random split...")

    np.random.seed(seed)
    interactions_df = interactions_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(interactions_df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_df = interactions_df.iloc[:n_test]
    val_df = interactions_df.iloc[n_test:n_test + n_val]
    train_df = interactions_df.iloc[n_test + n_val:]

    return train_df, val_df, test_df


def leave_one_out_split(interactions_df):
    """
    Leave-one-out split: for each user, last interaction is test,
    second to last is validation, rest is train.
    """
    logger.info("Creating leave-one-out split...")

    # Sort by user and timestamp
    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

    train_data = []
    val_data = []
    test_data = []

    for user_id, group in interactions_df.groupby('user_id'):
        group = group.sort_values('timestamp')
        n = len(group)

        if n < 3:
            # Not enough data, put all in train
            train_data.append(group)
            continue

        test_data.append(group.iloc[-1:])
        val_data.append(group.iloc[-2:-1])
        train_data.append(group.iloc[:-2])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test splits")
    parser.add_argument("--config", type=str, default="configs/data_sources.yaml")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=['time_based', 'random', 'leave_one_out'])
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get paths
    processed_dir = Path(config.get('paths', {}).get('processed_dir', 'data/processed/all_beauty'))

    # Input file
    interactions_file = processed_dir / "interactions.csv"
    items_file = processed_dir / "items.csv"

    if not interactions_file.exists():
        logger.error(f"Interactions file not found: {interactions_file}")
        logger.error("Run extract_and_normalize.py first!")
        sys.exit(1)

    # Load interactions and items
    logger.info(f"Loading interactions from {interactions_file}")
    interactions_df = pd.read_csv(interactions_file)
    logger.info(f"Loaded {len(interactions_df)} interactions")

    # Filter to only include items that have metadata
    if items_file.exists():
        items_df = pd.read_csv(items_file)
        valid_items = set(items_df['item_id'].values)
        original_count = len(interactions_df)
        interactions_df = interactions_df[interactions_df['item_id'].isin(valid_items)]
        logger.info(f"Filtered to {len(interactions_df)} interactions with valid items (removed {original_count - len(interactions_df)})")

    # Get split parameters
    split_config = config.get('splits', {})
    strategy = args.strategy or split_config.get('strategy', 'time_based')
    test_ratio = args.test_ratio or split_config.get('test_ratio', 0.2)
    val_ratio = args.val_ratio or split_config.get('val_ratio', 0.1)
    seed = args.seed or config.get('seed', 42)

    # Create split
    if strategy == 'time_based':
        train_df, val_df, test_df = time_based_split(interactions_df, test_ratio, val_ratio)
    elif strategy == 'random':
        train_df, val_df, test_df = random_split(interactions_df, test_ratio, val_ratio, seed)
    elif strategy == 'leave_one_out':
        train_df, val_df, test_df = leave_one_out_split(interactions_df)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Save splits
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # Summary
    logger.info("\n=== Split Summary ===")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    logger.info(f"Val: {len(val_df)} interactions, {val_df['user_id'].nunique()} users")
    logger.info(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")

    # Calculate sparsity
    n_users = interactions_df['user_id'].nunique()
    n_items = interactions_df['item_id'].nunique()
    n_interactions = len(interactions_df)
    total_possible = n_users * n_items
    sparsity = 1 - (n_interactions / total_possible) if total_possible > 0 else 0

    logger.info(f"\nDataset stats:")
    logger.info(f"Users: {n_users}")
    logger.info(f"Items: {n_items}")
    logger.info(f"Interactions: {n_interactions}")
    logger.info(f"Sparsity: {sparsity:.4%}")

    print(f"\nSplits saved to {processed_dir}")


if __name__ == "__main__":
    main()
