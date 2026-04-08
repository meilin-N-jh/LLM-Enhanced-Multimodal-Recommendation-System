#!/usr/bin/env python3
"""
Extract and normalize Amazon Reviews 2023 All_Beauty dataset.

Converts raw JSONL data to standardized CSV format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import logging
import sys

# Setup logging
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


def load_jsonl(filepath, max_lines=None):
    """Load JSONL file safely."""
    data = []
    filepath = Path(filepath)
    logger.info(f"Loading {filepath}...")

    # Handle .gz files
    import gzip
    open_func = gzip.open if str(filepath).endswith('.gz') else open

    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {i}: {e}")
                continue

    logger.info(f"Loaded {len(data)} records")
    return data


def extract_interactions(review_data, config):
    """Extract user-item interactions with proper filtering."""
    logger.info("Extracting interactions...")

    max_users = config.get('processing', {}).get('max_users')
    max_interactions = config.get('processing', {}).get('max_interactions')
    min_interactions = config.get('processing', {}).get('min_interactions_per_user', 5)  # 改为5
    positive_threshold = config.get('processing', {}).get('positive_rating_threshold', 4)
    verified_only = config.get('processing', {}).get('verified_purchases_only', True)

    interactions_data = []

    for i, review in enumerate(review_data):
        if max_interactions and i >= max_interactions:
            break

        user_id = review.get('user_id')
        asin = review.get('parent_asin') or review.get('asin')  # 使用parent_asin匹配metadata
        rating = review.get('rating', 0)
        timestamp = review.get('timestamp', 0)

        if not user_id or not asin:
            continue

        # Filter verified purchases
        if verified_only and not review.get('verified_purchase', False):
            continue

        # Create label (implicit feedback)
        label = 1 if rating >= positive_threshold else 0

        # Only keep positive interactions
        if label != 1:
            continue

        # Review text
        review_title = review.get('title', '') or ''
        review_text = review.get('text', '') or ''

        interactions_data.append({
            'user_id': user_id,
            'item_id': asin,
            'rating': rating,
            'label': label,
            'timestamp': timestamp,
            'review_title': review_title[:500] if review_title else '',
            'review_text': review_text[:2000] if review_text else ''
        })

    interactions_df = pd.DataFrame(interactions_data)

    # Filter users with minimum interactions FIRST
    min_item_interactions = config.get('processing', {}).get('min_item_interactions', 2)  # 商品最少2次交互

    if len(interactions_df) > 0:
        # First filter by user count
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]

        # Then filter by item count (item must have at least min_item_interactions)
        item_counts = interactions_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(valid_items)]

        # Re-filter users after item filtering (some users may have lost all interactions)
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]

        # Sample users if needed
        if max_users:
            valid_users_list = list(valid_users)[:max_users]
            interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users_list)]

    logger.info(f"Extracted {len(interactions_df)} interactions from {interactions_df['user_id'].nunique()} users, {interactions_df['item_id'].nunique()} items")

    return interactions_df


def extract_items_from_interactions(interactions_df, meta_data, config):
    """Extract items that appear in interactions + get their metadata."""
    logger.info("Extracting items from interactions...")

    # Get items that have interactions
    interacting_items = set(interactions_df['item_id'].unique())
    logger.info(f"Items with interactions: {len(interacting_items)}")

    items_data = []
    bought_together_relations = []

    for item in meta_data:
        # Use parent_asin to match
        asin = item.get('parent_asin') or item.get('asin')
        if not asin:
            continue

        # Only keep items that appear in interactions
        if asin not in interacting_items:
            continue

        # Title
        title = item.get('title', '') or ''
        title = title[:500] if title else 'Unknown'

        # Description
        description = ''
        if item.get('description'):
            if isinstance(item['description'], list):
                description = ' '.join(item['description'])
            else:
                description = str(item['description'])
        if item.get('features'):
            if isinstance(item['features'], list):
                description += ' ' + ' '.join(item['features'])
            else:
                description += ' ' + str(item['features'])
        description = description[:2000]

        # Category
        categories = item.get('categories', [])
        if categories and isinstance(categories[0], list):
            category = ' > '.join(categories[0]) if categories[0] else 'Unknown'
        else:
            category = str(categories[0]) if categories else item.get('main_category', 'Unknown')
        category = category[:200] if category else 'Unknown'

        # Brand
        brand = item.get('store', '') or ''
        brand = brand[:100]

        # Price
        price = item.get('price')
        if price:
            price_str = str(price).replace('$', '').replace(',', '')
            try:
                price = float(price_str)
            except:
                price = None
        else:
            price = None

        # Image URL
        images = item.get('images', [])
        image_url = ''
        if images and len(images) > 0:
            image_url = images[0].get('large') or images[0].get('hi_res') or images[0].get('thumb', '')

        items_data.append({
            'item_id': asin,
            'title': title,
            'description': description,
            'category': category,
            'brand': brand,
            'price': price,
            'image_url': image_url,
            'image_path': f'images/{asin}.jpg'
        })

        # Extract bought_together relations
        bought_together = item.get('bought_together', [])
        if bought_together and isinstance(bought_together, list):
            for related_asin in bought_together:
                # Only keep relations where both items are in interactions
                if related_asin in interacting_items:
                    bought_together_relations.append({
                        'item_id': asin,
                        'related_item_id': related_asin,
                        'relation_type': 'bought_together'
                    })

    items_df = pd.DataFrame(items_data)
    items_df = items_df.drop_duplicates(subset=['item_id'])
    logger.info(f"Extracted {len(items_df)} items with metadata")

    return items_df, bought_together_relations


def build_also_bought_relations(interactions_df, config):
    """Build also_bought relations from co-purchase patterns."""
    logger.info("Building also_bought relations from interactions...")

    # Build user purchase history
    user_purchases = defaultdict(set)
    for _, row in interactions_df.iterrows():
        user_purchases[row['user_id']].add(row['item_id'])

    # Find co-purchased items (items bought together by same user)
    relations = defaultdict(lambda: defaultdict(int))

    for user_id, items in user_purchases.items():
        items_list = list(items)
        for i, item1 in enumerate(items_list):
            for item2 in items_list[i+1:]:
                relations[item1][item2] += 1
                relations[item2][item1] += 1

    # Keep co-purchases (threshold = 1 means any user bought both)
    relations_data = []
    for item1, related in relations.items():
        for item2, count in related.items():
            if count >= 1:
                relations_data.append({
                    'item_id': item1,
                    'related_item_id': item2,
                    'relation_type': 'also_bought'
                })

    logger.info(f"Built {len(relations_data)} also_bought relations")
    return relations_data


def create_image_manifest(items_df):
    """Create image manifest for filtered items."""
    logger.info("Creating image manifest...")

    manifest_data = []
    for _, row in items_df.iterrows():
        manifest_data.append({
            'item_id': row['item_id'],
            'image_url': row['image_url'],
            'local_image_path': row['image_path'],
            'download_status': 'pending' if row['image_url'] else 'no_url'
        })

    manifest_df = pd.DataFrame(manifest_data)
    logger.info(f"Created manifest for {len(manifest_df)} items")

    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Extract and normalize Amazon All_Beauty data")
    parser.add_argument("--config", type=str, default="configs/data_sources.yaml")
    parser.add_argument("--source_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config()
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Source directory
    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = Path(config.get('paths', {}).get('raw_source', '/home/g41/project/data/amazon_2023/raw'))

    # Output directory
    output_dir = Path(config.get('paths', {}).get('processed_dir', 'data/processed/all_beauty'))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")

    # Check for existing data
    meta_file = source_dir / "meta_categories" / "meta_All_Beauty.jsonl"
    review_file = source_dir / "review_categories" / "All_Beauty.jsonl"

    if not meta_file.exists():
        logger.error(f"Metadata file not found: {meta_file}")
        sys.exit(1)

    if not review_file.exists():
        logger.error(f"Review file not found: {review_file}")
        sys.exit(1)

    # Load raw data - load more reviews to get enough interactions
    max_items = config.get('processing', {}).get('max_items', 50000)
    max_interactions = config.get('processing', {}).get('max_interactions', 100000)

    meta_data = load_jsonl(meta_file, max_items)
    review_data = load_jsonl(review_file, max_interactions)

    # STEP 1: Extract interactions FIRST
    interactions_df = extract_interactions(review_data, config)
    logger.info(f"After filtering: {len(interactions_df)} interactions, {interactions_df['user_id'].nunique()} users")

    # STEP 2: Extract items ONLY from interacting items
    items_df, bought_together = extract_items_from_interactions(interactions_df, meta_data, config)

    # Update config for next step
    config['interactions_df'] = interactions_df
    config['items_df'] = items_df

    # Save items
    items_df.to_csv(output_dir / "items.csv", index=False)
    logger.info(f"Saved {len(items_df)} items to {output_dir / 'items.csv'}")

    # Save interactions
    interactions_df.to_csv(output_dir / "interactions.csv", index=False)
    logger.info(f"Saved {len(interactions_df)} interactions to {output_dir / 'interactions.csv'}")

    # STEP 3: Build also_bought from interactions (better coverage)
    also_bought = build_also_bought_relations(interactions_df, config)

    # Combine relations
    relations_data = bought_together + also_bought

    if relations_data:
        relations_df = pd.DataFrame(relations_data)
        relations_df = relations_df.drop_duplicates()
        relations_df.to_csv(output_dir / "item_relations.csv", index=False)
        logger.info(f"Saved {len(relations_df)} relations to {output_dir / 'item_relations.csv'}")
    else:
        pd.DataFrame(columns=['item_id', 'related_item_id', 'relation_type']).to_csv(
            output_dir / "item_relations.csv", index=False
        )
        logger.warning("No relations found, created empty file")

    # Create image manifest
    manifest_df = create_image_manifest(items_df)
    manifest_df.to_csv(output_dir / "image_manifest.csv", index=False)
    logger.info(f"Saved image manifest to {output_dir / 'image_manifest.csv'}")

    # Summary
    logger.info("\n=== Extraction Summary ===")
    logger.info(f"Items: {len(items_df)}")
    logger.info(f"Interactions: {len(interactions_df)}")
    logger.info(f"Users: {interactions_df['user_id'].nunique() if len(interactions_df) > 0 else 0}")
    logger.info(f"Relations: {len(relations_data)}")
    logger.info(f"Items with images: {(items_df['image_url'] != '').sum()}")

    # Calculate density
    n_users = interactions_df['user_id'].nunique()
    n_items = len(items_df)
    n_interactions = len(interactions_df)
    total_possible = n_users * n_items
    density = n_interactions / total_possible * 100 if total_possible > 0 else 0

    logger.info(f"Interaction density: {density:.4f}%")

    print(f"\nExtraction complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
