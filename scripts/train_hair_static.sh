#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ID=$1
DATE=0215

# Stage 1+2: Train head mesh + canonical hair (HairTrainer)
WORKSPACE=checkpoints/mega/$DATE/
VERSION=train_${ID}_b8_MeGA_hair_static

DEFAULT_PARAMS=./configs/nersemble/${ID}/static_hair.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'
