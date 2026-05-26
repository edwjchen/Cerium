#!/bin/bash


export DEVICE="cpu"
NGPUS=1
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.1-8B
MASTER_PORT=10808 \
DEIVCE="cpu" PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --master_port 10808 --nproc_per_node=$NGPUS \
  -m completion_packed $CHECKPOINT_DIR \
  --world_size $NGPUS \
  

