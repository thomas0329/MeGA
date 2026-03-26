#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DATE=custom_multi-frame

WORKSPACE=checkpoints/mega/$DATE/
VERSION=train_whiteBg_b8_MeGA_hair

DEFAULT_PARAMS=./configs/nersemble/custom/hair.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'
