#!/bin/bash
#SBATCH -A nlp
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH -o wv_op.txt 
#SBATCH --job-name=wv_train
#SBATCH --time=3-00:00:00


python3 run.py Electronics_5.json 2
