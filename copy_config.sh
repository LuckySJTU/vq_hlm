#!/bin/bash

set -x

RUN_NAME=0322_rsimvq_CD512_CS16_CN64_LR1E-3_BS1
MODEL_CONFIG_DIR=./conf/models/residualsimvq.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
TRAIN_CONFIG_DIR=./conf/hlm_train/train_config.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml
cp $TRAIN_CONFIG_DIR $OUTPUT_DIR/train_config.yaml