"""Data loading utilities."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""

    def __init__(self, interactions, user2idx, item2idx, relations=None):
        """Initialize dataset.

        Args:
            interactions: DataFrame with user_id, item_id columns
            user2idx: User ID to index mapping
            item2idx: Item ID to index mapping
            relations: Optional relations DataFrame
        """
        self.interactions = interactions
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.relations = relations

        # Build user history
        self.user_history = {}
        for user_id, group in interactions.groupby('user_id'):
            self.user_history[user_id] = set(group['item_id'].tolist())

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['user_id']
        item_id = row['item_id']

        return {
            'user_idx': self.user2idx.get(user_id, 0),
            'item_idx': self.item2idx.get(item_id, 0),
            'user_id': user_id,
            'item_id': item_id
        }


class BPRDataset(Dataset):
    """Dataset for BPR training with negative sampling."""

    def __init__(self, samples, user2idx, item2idx):
        """Initialize dataset.

        Args:
            samples: List of (user, pos_item, neg_item) tuples
            user2idx: User ID to index mapping
            item2idx: Item ID to index mapping
        """
        self.samples = samples
        self.user2idx = user2idx
        self.item2idx = item2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, pos_item, neg_item = self.samples[idx]

        return {
            'user_idx': self.user2idx[user],
            'pos_item_idx': self.item2idx[pos_item],
            'neg_item_idx': self.item2idx[neg_item]
        }


class InferenceDataset(Dataset):
    """Dataset for inference."""

    def __init__(self, user_ids, user2idx, exclude_items=None):
        """Initialize dataset.

        Args:
            user_ids: List of user IDs
            user2idx: User ID to index mapping
            exclude_items: Dict mapping user_id to set of items to exclude
        """
        self.user_ids = user_ids
        self.user2idx = user2idx
        self.exclude_items = exclude_items or {}

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return {
            'user_idx': self.user2idx[user_id],
            'user_id': user_id,
            'exclude_items': self.exclude_items.get(user_id, set())
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'item_idx': torch.tensor([b['item_idx'] for b in batch], dtype=torch.long)
    }


def collate_fn_bpr(batch):
    """Collate function for BPR dataset."""
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'pos_item_idx': torch.tensor([b['pos_item_idx'] for b in batch], dtype=torch.long),
        'neg_item_idx': torch.tensor([b['neg_item_idx'] for b in batch], dtype=torch.long)
    }


def collate_fn_inference(batch):
    """Collate function for inference."""
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'user_ids': [b['user_id'] for b in batch],
        'exclude_items': [b['exclude_items'] for b in batch]
    }
