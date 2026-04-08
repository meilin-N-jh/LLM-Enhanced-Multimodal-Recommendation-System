#!/bin/bash
# Full data setup pipeline for Amazon All_Beauty dataset

set -e

echo "======================================"
echo "Amazon All_Beauty Data Setup Pipeline"
echo "======================================"

# Check conda environment
if ! conda info --envs | grep -q "g41_project"; then
    echo "Error: g41_project conda environment not found"
    echo "Create with: conda create -n g41_project python=3.10"
    exit 1
fi

# Activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate g41_project

echo ""
echo "Step 1: Extract and Normalize Data"
echo "-----------------------------------"
python src/data_setup/extract_and_normalize.py

echo ""
echo "Step 2: Build Train/Val/Test Splits"
echo "-------------------------------------"
python src/data_setup/build_splits.py

echo ""
echo "Step 3: Validate Dataset"
echo "------------------------"
python src/data_setup/validate_dataset.py

echo ""
echo "Step 4: Generate Summary"
echo "------------------------"
python src/data_setup/summarize_dataset.py

echo ""
echo "Step 5: Optional Image Download"
echo "-------------------------------"
echo "For a real multimodal run, download product images next:"
echo "  python src/data_setup/download_images.py --data_dir data/processed/all_beauty --workers 64"

echo ""
echo "======================================"
echo "Data Setup Complete!"
echo "======================================"
echo ""
echo "Output files:"
echo "  data/processed/all_beauty/items.csv"
echo "  data/processed/all_beauty/interactions.csv"
echo "  data/processed/all_beauty/item_relations.csv"
echo "  data/processed/all_beauty/image_manifest.csv"
echo "  data/processed/all_beauty/train.csv"
echo "  data/processed/all_beauty/val.csv"
echo "  data/processed/all_beauty/test.csv"
echo "  data/processed/all_beauty/dataset_summary.json"
