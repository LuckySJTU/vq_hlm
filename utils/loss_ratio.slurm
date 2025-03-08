#!/bin/bash

#SBATCH --job-name=loss_ratio
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err
#SBATCH --partition=ADA6000,L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:0

cd /home/yxwang/vq_hlm/utils
python loss_ratio_gaussian.py