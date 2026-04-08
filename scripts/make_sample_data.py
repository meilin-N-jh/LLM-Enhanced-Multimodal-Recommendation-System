#!/usr/bin/env python3
"""Generate sample data for the recommendation system."""

import os
import csv
import random
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Configuration
N_USERS = 100
N_ITEMS = 50
N_INTERACTIONS = 500
N_RELATIONS = 80

DATA_DIR = Path("data/sample")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Generate items
print("Generating items...")
items = []
categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

for i in range(1, N_ITEMS + 1):
    item_id = f"i{i}"
    title = f"Product {i}"
    description = f"This is a great product number {i} with excellent features."
    category = random.choice(categories)
    brand = random.choice(brands)
    # Use placeholder path (no actual images needed for sample)
    image_path = f"images/item_{i}.jpg"

    items.append({
        "item_id": item_id,
        "title": title,
        "description": description,
        "category": category,
        "brand": brand,
        "image_path": image_path
    })

items_df = pd.DataFrame(items)
items_df.to_csv(DATA_DIR / "items.csv", index=False)
print(f"Generated {len(items_df)} items")

# Generate user-item interactions
print("Generating interactions...")
interactions = []
used_pairs = set()

# Ensure each user has at least 3 interactions
for u in range(1, N_USERS + 1):
    user_id = f"u{u}"
    # First add some guaranteed interactions
    n_interactions = max(3, int(random.gauss(N_INTERACTIONS // N_USERS, 2)))

    for _ in range(n_interactions):
        item_id = f"i{random.randint(1, N_ITEMS)}"
        pair = (user_id, item_id)

        if pair not in used_pairs:
            used_pairs.add(pair)
            timestamp = random.randint(1000, 2000)
            interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "label": 1,
                "timestamp": timestamp
            })

# Convert to DataFrame and sort by timestamp
interactions_df = pd.DataFrame(interactions)
interactions_df = interactions_df.sort_values("timestamp")
interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)
print(f"Generated {len(interactions_df)} interactions")

# Generate item-item relations
print("Generating item relations...")
relations = []
relation_types = ["also_bought", "also_viewed"]

for _ in range(N_RELATIONS):
    item_id = f"i{random.randint(1, N_ITEMS)}"
    related_item_id = f"i{random.randint(1, N_ITEMS)}"

    if item_id != related_item_id:
        relations.append({
            "item_id": item_id,
            "related_item_id": related_item_id,
            "relation_type": random.choice(relation_types)
        })

relations_df = pd.DataFrame(relations)
relations_df.to_csv(DATA_DIR / "item_relations.csv", index=False)
print(f"Generated {len(relations_df)} relations")

# Create images directory placeholder
images_dir = DATA_DIR / "images"
images_dir.mkdir(exist_ok=True)

# Create a placeholder README in images dir
(images_dir / "README.txt").write_text(
    "Place item images here. Image filenames should match items.csv image_path.\n"
    "Missing images will use zero vectors as fallback.\n"
)

print("\nSample data generated successfully!")
print(f"  - Items: {DATA_DIR / 'items.csv'}")
print(f"  - Interactions: {DATA_DIR / 'interactions.csv'}")
print(f"  - Relations: {DATA_DIR / 'item_relations.csv'}")
print("\nTo extract features, run: python scripts/extract_features.py")
