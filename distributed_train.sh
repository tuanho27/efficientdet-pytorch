#!/bin/bash
NUM_PROC=$1
shift
export CUDA_VISIBLE_DEVICES=0,1;
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC training_efficientdet.py "$@" 