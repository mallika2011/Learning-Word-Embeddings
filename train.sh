#!/bin/bash
#SBATCH -A research
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END
#SBATCH -o wv_op.txt 
#SBATCH --job-name=wv_train
#SBATCH --time=3-00:00:00


python3 run.py Electronics_5.json 0
