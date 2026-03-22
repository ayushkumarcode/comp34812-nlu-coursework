#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=av_cat_c
#SBATCH --output=logs/av_cat_c_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

cd ~/scratch/nlu-project
python -u -m src.training.train_cat_c --task av --batch_size 16 --epochs 25
