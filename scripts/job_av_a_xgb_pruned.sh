#!/bin/bash
#SBATCH -p multicore
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name=av_a_xgb
#SBATCH --output=logs/av_a_xgb_pruned_%j.out

cd ~/scratch/nlu-project
module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/av_cat_a_xgb_pruned.py
