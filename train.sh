#!/bin/bash

set -x

RUN_NAME=1219_lowdim_CBDIM8
MODEL_CONFIG_DIR=./conf/models/vectorquantize.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml

python train_vq.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR 2>&1 |tee $OUTPUT_DIR/train.log

# test
# python train_vq.py --data_config $DATA_CONFIG_DIR --model_config $MODEL_CONFIG_DIR --ckpt_dir $OUTPUT_DIR --test
