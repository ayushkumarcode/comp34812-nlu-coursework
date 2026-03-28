#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH -o logs/infer_nli_c_%j.out
#SBATCH -e logs/infer_nli_c_%j.err

cd ~/scratch/nlu-project
mkdir -p logs predictions

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/infer_nli_cat_c.py
