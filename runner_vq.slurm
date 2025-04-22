#!/bin/bash

#SBATCH --job-name=vqhlm
#SBATCH --partition=ADA6000,L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err

bash train.sh $1