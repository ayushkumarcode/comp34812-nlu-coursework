#!/bin/bash
#SBATCH -p multicore
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=av_a_novel
#SBATCH --output=logs/av_a_novel_%j.out
#SBATCH --error=logs/av_a_novel_%j.err

set -e
echo "Job started on $(hostname) at $(date)"

module load apps/binapps/conda/miniforge3/25.3.0
source activate nlu

echo "Conda env activated, python=$(which python)"

cd ~/scratch/nlu-project
export PYTHONUNBUFFERED=1
python -u scripts/retrain_av_a_augment_features.py

echo "Job finished at $(date)"
