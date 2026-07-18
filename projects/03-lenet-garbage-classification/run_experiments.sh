#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

python train_mnist.py --epochs 2 --train-limit 20000 --test-limit 5000
python finetune_garbage.py --strategy scratch --epochs 8 --lr 0.001
python finetune_garbage.py --strategy frozen --epochs 8 --lr 0.001
python finetune_garbage.py --strategy full --epochs 8 --lr 0.0005
