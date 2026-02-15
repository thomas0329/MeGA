#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ID=$1
DATE=0215

# Stage 3: Joint fine-tuning with static mode (JointTrainer)
WORKSPACE=checkpoints/mega/$DATE/
VERSION=train_${ID}_b16_MeGA_full_static

DEFAULT_PARAMS=./configs/nersemble/${ID}/static_full.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'
