#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --job-name=av_c_xenc_v2
#SBATCH --output=logs/av_c_crossenc_v2_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

cd ~/scratch/nlu-project
python -u scripts/train_av_cat_c_crossenc_v2.py
