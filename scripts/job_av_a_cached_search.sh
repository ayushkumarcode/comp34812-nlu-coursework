#!/bin/bash
#SBATCH -p multicore
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --job-name=av_a_cs
#SBATCH --output=logs/av_a_cached_search_%j.out
#SBATCH --dependency=afterok:12651829

cd ~/scratch/nlu-project
module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/av_cat_a_cached_search.py
