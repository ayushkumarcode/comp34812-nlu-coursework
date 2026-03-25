#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=av_c_ext
#SBATCH --output=logs/extract_av_c_probs_%j.out

cd ~/scratch/nlu-project
module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/extract_av_c_probs.py
