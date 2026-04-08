"""Data preprocessing functions."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir):
    """Load all data files."""
    data_dir = Path(data_dir)

    interactions = pd.read_csv(data_dir / "interactions.csv")
    items = pd.read_csv(data_dir / "items.csv")
    relations = pd.read_csv(data_dir / "item_relations.csv")

    return interactions, items, relations


def preprocess_interactions(interactions_df, min_interactions=1):
    """Preprocess interactions data.

    Args:
        interactions_df: DataFrame with user_id, item_id, label, timestamp
        min_interactions: Minimum interactions per user

    Returns:
        Filtered interactions DataFrame
    """
    # Filter by label (positive interactions)
    df = interactions_df[interactions_df['label'] > 0].copy()

    # Filter users with minimum interactions
    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)]

    # Sort by timestamp
    df = df.sort_values(['user_id', 'timestamp'])

    return df


def create_user_item_mapping(df):
    """Create user and item ID mappings."""
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())

    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2user = {i: u for u, i in user2idx.items()}

    item2idx = {it: i for i, it in enumerate(unique_items)}
    idx2item = {i: it for it, i in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item


def split_data(interactions, strategy="time_based", test_ratio=0.2, val_ratio=0.1):
    """Split data into train/val/test.

    Args:
        interactions: Preprocessed interactions DataFrame
        strategy: "time_based" or "leave_one_out"
        test_ratio: Ratio of data for test
        val_ratio: Ratio of data for validation

    Returns:
        train_df, val_df, test_df
    """
    if strategy == "time_based":
        # Sort by timestamp
        interactions = interactions.sort_values('timestamp')

        # Split by ratio
        n = len(interactions)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        test_df = interactions.iloc[-test_size:]
        val_df = interactions.iloc[-(test_size + val_size):-test_size]
        train_df = interactions.iloc[:-(test_size + val_size)]

    elif strategy == "leave_one_out":
        # Leave one out: last interaction per user is test
        train_df = []
        test_df = []

        for user_id, group in interactions.groupby('user_id'):
            group = group.sort_values('timestamp')
            if len(group) > 1:
                train_df.append(group.iloc[:-1])
                test_df.append(group.iloc[-1:])
            else:
                train_df.append(group)

        train_df = pd.concat(train_df)
        test_df = pd.concat(test_df)
        val_df = pd.DataFrame(columns=interactions.columns)

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return train_df, val_df, test_df


def create_negative_samples(
    interactions_df,
    item_ids,
    n_neg=1,
    seed=42,
    strategy="uniform",
    relation_lookup=None,
    hard_negative_ratio=0.0,
    popular_negative_ratio=0.0,
    popularity_alpha=1.0,
):
    """Create negative samples for training.

    Args:
        interactions_df: Training interactions
        item_ids: List of all item IDs
        n_neg: Number of negative samples per positive
        seed: Random seed
        strategy: uniform, popularity, or mixed
        relation_lookup: Optional item -> related items lookup
        hard_negative_ratio: Probability of sampling a relation-based hard negative
        popular_negative_ratio: Probability of sampling by popularity
        popularity_alpha: Temperature for popularity weights

    Returns:
        List of (user, positive_item, negative_item) tuples
    """
    rng = np.random.default_rng(seed)
    item_ids = np.asarray(list(item_ids), dtype=object)
    valid_item_set = set(item_ids.tolist())

    # Build positive interaction set
    positive_items = set(zip(interactions_df['user_id'], interactions_df['item_id']))
    user_items = interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
    relation_lookup = relation_lookup or {}

    popularity_probs = None
    if strategy in {"popularity", "mixed"} or popular_negative_ratio > 0:
        item_counts = interactions_df['item_id'].value_counts()
        popularity_scores = np.array(
            [float(item_counts.get(item_id, 1.0)) ** popularity_alpha for item_id in item_ids],
            dtype=np.float64,
        )
        popularity_probs = popularity_scores / popularity_scores.sum()

    def sample_uniform():
        return rng.choice(item_ids)

    def sample_popular():
        if popularity_probs is None:
            return sample_uniform()
        return rng.choice(item_ids, p=popularity_probs)

    def draw_negative(user, pos_item):
        user_positive = user_items.get(user, set())

        if strategy == "mixed" and hard_negative_ratio > 0 and rng.random() < hard_negative_ratio:
            hard_pool = [
                item_id for item_id in relation_lookup.get(pos_item, set())
                if item_id in valid_item_set and item_id not in user_positive and item_id != pos_item
            ]
            if hard_pool:
                return rng.choice(np.asarray(hard_pool, dtype=object))

        if strategy == "popularity" or (strategy == "mixed" and popular_negative_ratio > 0 and rng.random() < popular_negative_ratio):
            for _ in range(64):
                neg_item = sample_popular()
                if (user, neg_item) not in positive_items:
                    return neg_item

        for _ in range(64):
            neg_item = sample_uniform()
            if (user, neg_item) not in positive_items:
                return neg_item

        fallback_pool = [item_id for item_id in item_ids if item_id not in user_positive]
        if fallback_pool:
            return rng.choice(np.asarray(fallback_pool, dtype=object))
        return pos_item

    samples = []
    for _, row in interactions_df.iterrows():
        user = row['user_id']
        pos_item = row['item_id']

        for _ in range(n_neg):
            neg_item = draw_negative(user, pos_item)
            samples.append((user, pos_item, neg_item))

    return samples
