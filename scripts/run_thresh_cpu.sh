#!/bin/bash
#SBATCH -p serial
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --job-name=thresh_cpu
#SBATCH --output=logs/thresh_cpu_%j.log

module load apps/binapps/conda/miniforge3/25.3.0
conda activate nlu
cd ~/scratch/nlu-project

echo "=== NLI Cat A threshold optimization ==="
python -u scripts/get_probs_nli_a.py

echo ""
echo "NLI Cat A threshold job done."
