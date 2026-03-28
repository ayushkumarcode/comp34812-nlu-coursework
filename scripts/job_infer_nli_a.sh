#!/bin/bash
#SBATCH -p serial
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH -o logs/infer_nli_a_%j.out
#SBATCH -e logs/infer_nli_a_%j.err

cd ~/scratch/nlu-project
mkdir -p logs predictions

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

python -u scripts/infer_nli_cat_a.py
