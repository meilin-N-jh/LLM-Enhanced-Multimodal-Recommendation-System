#!/bin/bash
# Training script for the recommendation system

set -e

# Default config
CONFIG="configs/default.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --no_image)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_image"
            shift
            ;;
        --no_text)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_text"
            shift
            ;;
        --no_relation)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_relation"
            shift
            ;;
        --no_rerank)
            EXTRA_FLAGS="$EXTRA_FLAGS --no_rerank"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting training with config: $CONFIG"
echo "Extra flags: $EXTRA_FLAGS"

python -m src.trainer --config "$CONFIG" $EXTRA_FLAGS

echo "Training complete!"
