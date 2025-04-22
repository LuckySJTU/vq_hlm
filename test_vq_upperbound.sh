#!/bin/bash

set -x

RUN_NAME=$1
OUTPUT_DIR=./exp/$RUN_NAME

python train_vq_upperbound.py \
--vq_dir $OUTPUT_DIR \
--test \
2>&1 | tee -a $OUTPUT_DIR/testhlm.log