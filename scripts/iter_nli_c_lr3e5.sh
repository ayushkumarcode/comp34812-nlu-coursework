#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=nli_c_lr3e5
#SBATCH --output=logs/nli_c_lr3e5_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu

cd ~/scratch/nlu-project
python -u -m src.training.train_cat_c --task nli --batch_size 32 --epochs 15 --lr 3e-5
