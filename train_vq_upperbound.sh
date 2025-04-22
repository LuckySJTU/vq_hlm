#!/bin/bash

set -x

RUN_NAME=$1

OUTPUT_DIR=./exp/$RUN_NAME

torchrun --nproc_per_node=4 --master_port 11427 train_vq_upperbound.py \
--vq_dir $OUTPUT_DIR \
2>&1 | tee $OUTPUT_DIR/trainhlm.log