#!/bin/bash

set -x

RUN_NAME=$1
OUTPUT_DIR=./exp/$RUN_NAME

python train_hlm_gpt2.py \
--vq_dir $OUTPUT_DIR \
--test \
| tee $OUTPUT_DIR/testhlm.log