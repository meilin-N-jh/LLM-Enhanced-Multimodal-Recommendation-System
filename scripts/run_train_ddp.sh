#!/bin/bash
# Distributed training with 8 GPUs

CONFIG="configs/default.yaml"
GPUS=8

echo "Starting distributed training with $GPUS GPUs"

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    src/trainer_ddp.py \
    --config "$CONFIG"

echo "Training complete!"
