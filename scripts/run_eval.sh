#!/bin/bash
# Evaluation script

set -e

# Default config
CONFIG="configs/default.yaml"
MODEL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find latest model if not specified
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls -t outputs/checkpoints/*.pt 2>/dev/null | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model found. Please train first or specify --model"
        exit 1
    fi
fi

echo "Evaluating model: $MODEL_PATH"

python -m src.evaluator \
    --config "$CONFIG" \
    --model_path "$MODEL_PATH"

echo "Evaluation complete!"
