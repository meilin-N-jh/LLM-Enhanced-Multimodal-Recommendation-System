#!/usr/bin/env python3
"""
Generate dataset summary statistics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

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


def compute_summary(processed_dir, config):
    """Compute comprehensive dataset summary."""
    logger.info("Computing dataset summary...")

    summary = {
        'source': {
            'category': config.get('category', 'All_Beauty'),
            'source_url': 'Amazon Reviews 2023',
            'raw_data_path': str(config.get('paths', {}).get('raw_source', ''))
        }
    }

    # Load data
    items_file = processed_dir / "items.csv"
    interactions_file = processed_dir / "interactions.csv"
    relations_file = processed_dir / "item_relations.csv"
    image_manifest_file = processed_dir / "image_manifest.csv"

    # Items summary
    if items_file.exists():
        items_df = pd.read_csv(items_file)
        summary['items'] = {
            'total': len(items_df),
            'with_title': (items_df['title'].notna() & (items_df['title'] != '')).sum(),
            'with_description': (items_df['description'].notna() & (items_df['description'] != '')).sum(),
            'with_brand': (items_df['brand'].notna() & (items_df['brand'] != '')).sum(),
            'with_price': items_df['price'].notna().sum(),
            'with_image_url': (items_df['image_url'].notna() & (items_df['image_url'] != '')).sum(),
            'unique_categories': items_df['category'].nunique()
        }

        # Top categories
        if 'category' in items_df.columns:
            top_cats = items_df['category'].value_counts().head(10).to_dict()
            summary['items']['top_categories'] = top_cats

    # Interactions summary
    if interactions_file.exists():
        interactions_df = pd.read_csv(interactions_file)

        n_users = interactions_df['user_id'].nunique()
        n_items = interactions_df['item_id'].nunique()
        n_interactions = len(interactions_df)
        total_possible = n_users * n_items

        summary['interactions'] = {
            'total': n_interactions,
            'unique_users': n_users,
            'unique_items': n_items,
            'sparsity': float(1 - n_interactions / total_possible) if total_possible > 0 else 0,
            'avg_interactions_per_user': n_interactions / n_users if n_users > 0 else 0,
            'avg_interactions_per_item': n_interactions / n_items if n_items > 0 else 0
        }

        # Rating distribution
        if 'rating' in interactions_df.columns:
            summary['interactions']['rating_distribution'] = \
                interactions_df['rating'].value_counts().to_dict()

        # Label distribution
        if 'label' in interactions_df.columns:
            summary['interactions']['label_distribution'] = \
                interactions_df['label'].value_counts().to_dict()

        # Timestamp range
        if 'timestamp' in interactions_df.columns:
            summary['interactions']['timestamp_range'] = {
                'min': int(interactions_df['timestamp'].min()),
                'max': int(interactions_df['timestamp'].max())
            }

    # Relations summary
    if relations_file.exists():
        relations_df = pd.read_csv(relations_file)

        if len(relations_df) > 0:
            summary['relations'] = {
                'total': len(relations_df),
                'by_type': relations_df['relation_type'].value_counts().to_dict()
            }
        else:
            summary['relations'] = {
                'total': 0,
                'by_type': {}
            }

    # Image manifest summary
    if image_manifest_file.exists():
        manifest_df = pd.read_csv(image_manifest_file)

        summary['images'] = {
            'total': len(manifest_df),
            'with_url': (manifest_df['image_url'].notna() & (manifest_df['image_url'] != '')).sum(),
            'pending_download': (manifest_df['download_status'] == 'pending').sum()
        }

    # Splits summary
    splits_summary = {}
    for split_name in ['train', 'val', 'test']:
        split_file = processed_dir / f"{split_name}.csv"
        if split_file.exists():
            split_df = pd.read_csv(split_file)
            splits_summary[split_name] = {
                'interactions': len(split_df),
                'unique_users': int(split_df['user_id'].nunique()),
                'unique_items': int(split_df['item_id'].nunique())
            }

    if splits_summary:
        summary['splits'] = splits_summary

    # Missing data summary
    if items_file.exists():
        items_df = pd.read_csv(items_file)
        missing_summary = {}

        for col in ['title', 'description', 'brand', 'price', 'image_url']:
            if col in items_df.columns:
                missing = items_df[col].isna().sum() + (items_df[col] == '').sum()
                missing_summary[col] = int(missing)

        summary['missing_data'] = missing_summary

    # Processing config
    summary['processing_config'] = {
        'max_items': config.get('processing', {}).get('max_items'),
        'max_users': config.get('processing', {}).get('max_users'),
        'max_interactions': config.get('processing', {}).get('max_interactions'),
        'min_interactions_per_user': config.get('processing', {}).get('min_interactions_per_user'),
        'positive_rating_threshold': config.get('processing', {}).get('positive_rating_threshold'),
        'split_strategy': config.get('splits', {}).get('strategy')
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate dataset summary")
    parser.add_argument("--config", type=str, default="configs/data_sources.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get paths
    processed_dir = Path(config.get('paths', {}).get('processed_dir', 'data/processed/all_beauty'))

    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        logger.error("Run extract_and_normalize.py first!")
        return

    # Compute summary
    summary = compute_summary(processed_dir, config)

    # Print summary
    logger.info("\n=== Dataset Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    # Save to file
    output_file = Path(args.output) if args.output else processed_dir / "dataset_summary.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
