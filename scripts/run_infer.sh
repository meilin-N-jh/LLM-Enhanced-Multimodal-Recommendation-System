#!/bin/bash
# Inference script for recommendations

set -e

CONFIG="configs/default.yaml"
MODEL_PATH=""
USER_ID=""
TOP_N=10

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
        --user)
            USER_ID="$2"
            shift 2
            ;;
        --top_n)
            TOP_N="$2"
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

# Default user if not specified
if [ -z "$USER_ID" ]; then
    USER_ID="AE22XHMBOBJBXUFCTNYLFMD4UKMA"
fi

echo "Running inference for user: $USER_ID"
echo "Model: $MODEL_PATH"
echo "Top N: $TOP_N"

python -m src.inference \
    --config "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --user_id "$USER_ID" \
    --top_n "$TOP_N"

echo "Inference complete!"
