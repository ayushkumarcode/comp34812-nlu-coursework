#!/bin/bash
#SBATCH -p serial
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --job-name=av_cache
#SBATCH --output=logs/cache_av_features_%j.out

cd ~/scratch/nlu-project
module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/cache_av_features.py
