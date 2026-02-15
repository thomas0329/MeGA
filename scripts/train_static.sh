#!/bin/bash

# Run full static pipeline (head + canonical hair + joint fine-tuning)
bash ./scripts/train_hair_static.sh $1
bash ./scripts/train_full_static.sh $1
