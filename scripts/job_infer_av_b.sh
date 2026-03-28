#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH -o logs/infer_av_b_%j.out
#SBATCH -e logs/infer_av_b_%j.err

cd ~/scratch/nlu-project
mkdir -p logs predictions

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/infer_av_cat_b.py
