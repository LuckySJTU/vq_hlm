#!/bin/bash

#SBATCH --job-name=vq_upperbound
#SBATCH --partition=A100,ADA6000,L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=luckywang@sjtu.edu.cn

export NCCL_P2P_DISABLE=1
bash train_vq_upperbound.sh $1
bash test_vq_upperbound.sh $1