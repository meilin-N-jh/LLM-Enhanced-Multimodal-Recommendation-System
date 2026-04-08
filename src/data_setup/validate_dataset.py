#!/usr/bin/env python3
"""
Validate the processed dataset for completeness and correctness.
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


def validate_interactions(interactions_df, items_df):
    """Validate interactions file."""
    logger.info("Validating interactions...")

    errors = []
    warnings = []

    # Check required columns
    required_cols = ['user_id', 'item_id', 'rating', 'label', 'timestamp']
    missing_cols = set(required_cols) - set(interactions_df.columns)
    if missing_cols:
        errors.append(f"Missing columns in interactions: {missing_cols}")

    # Check for nulls
    null_counts = interactions_df[['user_id', 'item_id']].isnull().sum()
    if null_counts.any():
        errors.append(f"Null values in interactions: {null_counts[null_counts > 0].to_dict()}")

    # Check unique users/items
    n_users = interactions_df['user_id'].nunique()
    n_items = interactions_df['item_id'].nunique()
    logger.info(f"  Users: {n_users}, Items: {n_items}")

    # Check item coverage
    interaction_items = set(interactions_df['item_id'].unique())
    all_items = set(items_df['item_id'].values)
    uncovered_items = all_items - interaction_items

    if len(uncovered_items) > len(all_items) * 0.5:
        warnings.append(f"Only {len(interaction_items)}/{len(all_items)} items have interactions")

    # Check rating range
    if 'rating' in interactions_df.columns:
        if not interactions_df['rating'].between(1, 5).all():
            warnings.append("Some ratings are outside 1-5 range")

    # Check label values
    if 'label' in interactions_df.columns:
        unique_labels = interactions_df['label'].unique()
        if not all(l in [0, 1] for l in unique_labels):
            errors.append(f"Invalid label values: {unique_labels}")

    # Check for duplicates
    duplicates = interactions_df.duplicated(subset=['user_id', 'item_id']).sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate user-item pairs")

    return errors, warnings


def validate_items(items_df):
    """Validate items file."""
    logger.info("Validating items...")

    errors = []
    warnings = []

    # Check required columns
    required_cols = ['item_id', 'title', 'category']
    missing_cols = set(required_cols) - set(items_df.columns)
    if missing_cols:
        errors.append(f"Missing columns in items: {missing_cols}")

    # Check for null item_id
    null_count = items_df['item_id'].isnull().sum()
    if null_count > 0:
        errors.append(f"Found {null_count} null item_ids")

    # Check duplicates
    duplicates = items_df['item_id'].duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate item_ids")

    # Check for missing titles
    missing_titles = (items_df['title'].isnull() | (items_df['title'] == '')).sum()
    if missing_titles > len(items_df) * 0.3:
        warnings.append(f"{missing_titles} items have missing titles")

    # Check for missing descriptions
    missing_desc = (items_df['description'].isnull() | (items_df['description'] == '')).sum()
    logger.info(f"  Items with description: {len(items_df) - missing_desc}/{len(items_df)}")

    # Check for missing brands
    missing_brand = (items_df['brand'].isnull() | (items_df['brand'] == '')).sum()
    logger.info(f"  Items with brand: {len(items_df) - missing_brand}/{len(items_df)}")

    # Check images
    has_image = (items_df['image_url'] != '').sum()
    logger.info(f"  Items with image URL: {has_image}/{len(items_df)}")

    return errors, warnings


def validate_relations(relations_df, items_df):
    """Validate relations file."""
    logger.info("Validating relations...")

    errors = []
    warnings = []

    if len(relations_df) == 0:
        warnings.append("No relations found")
        return errors, warnings

    # Check required columns
    required_cols = ['item_id', 'related_item_id', 'relation_type']
    missing_cols = set(required_cols) - set(relations_df.columns)
    if missing_cols:
        errors.append(f"Missing columns in relations: {missing_cols}")

    # Check valid item_ids
    valid_items = set(items_df['item_id'].values)

    invalid_from = ~relations_df['item_id'].isin(valid_items)
    invalid_to = ~relations_df['related_item_id'].isin(valid_items)

    if invalid_from.any():
        warnings.append(f"{invalid_from.sum()} relations have invalid source items")

    if invalid_to.any():
        warnings.append(f"{invalid_to.sum()} relations have invalid target items")

    # Check relation types
    valid_types = {'also_bought', 'also_viewed', 'bought_together'}
    found_types = set(relations_df['relation_type'].unique())
    invalid_types = found_types - valid_types
    if invalid_types:
        warnings.append(f"Unknown relation types: {invalid_types}")

    logger.info(f"  Total relations: {len(relations_df)}")
    logger.info(f"  Relation types: {found_types}")

    return errors, warnings


def validate_splits(processed_dir):
    """Validate train/val/test splits."""
    logger.info("Validating splits...")

    errors = []
    warnings = []

    for split_name in ['train', 'val', 'test']:
        split_file = processed_dir / f"{split_name}.csv"
        if not split_file.exists():
            errors.append(f"Missing {split_name}.csv")
            continue

        split_df = pd.read_csv(split_file)
        logger.info(f"  {split_name}: {len(split_df)} interactions")

        # Check for required columns
        if 'user_id' not in split_df.columns or 'item_id' not in split_df.columns:
            errors.append(f"{split_name}.csv missing required columns")

    # Check user/item overlap
    try:
        train_df = pd.read_csv(processed_dir / "train.csv")
        test_df = pd.read_csv(processed_dir / "test.csv")

        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())

        # All test users should be in train
        missing_users = test_users - train_users
        if len(missing_users) > 0:
            warnings.append(f"{len(missing_users)} test users not in train (cold start)")

    except Exception as e:
        logger.warning(f"Could not validate user overlap: {e}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate processed dataset")
    parser.add_argument("--config", type=str, default="configs/data_sources.yaml")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings too")
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get paths
    processed_dir = Path(config.get('paths', {}).get('processed_dir', 'data/processed/all_beauty'))

    logger.info(f"Validating dataset in: {processed_dir}")

    all_errors = []
    all_warnings = []

    # Check if data exists
    items_file = processed_dir / "items.csv"
    interactions_file = processed_dir / "interactions.csv"

    if not items_file.exists():
        logger.error(f"Items file not found: {items_file}")
        logger.error("Run extract_and_normalize.py first!")
        sys.exit(1)

    if not interactions_file.exists():
        logger.error(f"Interactions file not found: {interactions_file}")
        logger.error("Run extract_and_normalize.py first!")
        sys.exit(1)

    # Load data
    items_df = pd.read_csv(items_file)
    interactions_df = pd.read_csv(interactions_file)

    # Validate
    errors, warnings = validate_interactions(interactions_df, items_df)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    errors, warnings = validate_items(items_df)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # Validate relations if exists
    relations_file = processed_dir / "item_relations.csv"
    if relations_file.exists():
        relations_df = pd.read_csv(relations_file)
        errors, warnings = validate_relations(relations_df, items_df)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    # Validate splits if exists
    splits_ok = True
    for split_name in ['train', 'val', 'test']:
        if not (processed_dir / f"{split_name}.csv").exists():
            splits_ok = False
            break

    if splits_ok:
        errors, warnings = validate_splits(processed_dir)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    else:
        all_warnings.append("Splits not found - run build_splits.py")

    # Report
    logger.info("\n=== Validation Results ===")

    if all_errors:
        logger.error("ERRORS:")
        for err in all_errors:
            logger.error(f"  - {err}")

    if all_warnings:
        logger.warning("WARNINGS:")
        for warn in all_warnings:
            logger.warning(f"  - {warn}")

    if not all_errors and not all_warnings:
        logger.info("Dataset is valid!")
    elif not all_errors:
        logger.info("Dataset is valid with warnings")
    else:
        logger.error("Dataset has errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
