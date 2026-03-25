#!/bin/bash
#SBATCH -p gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=thresh_gpu
#SBATCH --output=logs/thresh_gpu_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu
cd ~/scratch/nlu-project

echo "=== NLI Cat C threshold optimization ==="
python -u scripts/get_probs_nli_c.py

echo ""
echo "=== AV Cat C threshold optimization ==="
python -u scripts/get_probs_av_c.py

echo ""
echo "All GPU threshold jobs done."
