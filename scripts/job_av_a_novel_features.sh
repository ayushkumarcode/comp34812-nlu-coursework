#!/bin/bash
#SBATCH -p serial
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --job-name=av_a_novel
#SBATCH --output=logs/av_a_novel_%j.out

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

cd ~/scratch/nlu-project
python scripts/retrain_av_a_novel_features.py
