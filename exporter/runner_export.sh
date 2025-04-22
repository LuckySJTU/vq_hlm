#!/bin/bash

#SBATCH --job-name=openwebtext_export
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --mem=150G
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=10  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2

echo doing $1

MODEL=/data1/public/hf/openai-community/gpt2
# available splits: train/validation/test
python -u export_hs.py \
  --model_name_or_path ${MODEL} \
  --dataset_name /data1/public/hf/iohadrubin/wikitext-103-raw-v1/ \
  --output_dir /data0/yxwang/Dataset/vqhlm/wikitext103_gpt2/ \
  --do_eval --eval_subset $1 \
  --stride 1024 \
  --local_rank -1