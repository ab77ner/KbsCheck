#!/usr/bin/env bash
set -e
python train.py --config configs/default_config.yaml
python evaluate.py --ckpt outputs/best_model.pt --data data/test.jsonl
