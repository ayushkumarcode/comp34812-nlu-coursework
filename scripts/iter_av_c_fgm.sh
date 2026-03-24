#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=av_c_fgm
#SBATCH --output=logs/av_c_fgm_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu
cd ~/scratch/nlu-project
python -u scripts/train_av_c_fgm.py
