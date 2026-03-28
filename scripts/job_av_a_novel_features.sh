#!/bin/bash
#SBATCH -p multicore
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=av_a_novel
#SBATCH --output=logs/av_a_novel_%j.out

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

cd ~/scratch/nlu-project
export PYTHONUNBUFFERED=1
python -u scripts/retrain_av_a_novel_features.py
