#!/bin/bash
# Assemble final submission zip
# Usage: bash scripts/assemble_submission.sh [av|nli]

TASK=${1:-av}
echo "Assembling submission for track: $TASK"

# Create submission directory
rm -rf submission
mkdir -p submission

# Copy prediction files
cp predictions/Group_34_A.csv submission/
cp predictions/Group_34_B.csv submission/

# Copy model cards
cp model_cards/model_card_cat_a.md submission/
cp model_cards/model_card_cat_b.md submission/

# Copy notebooks
mkdir -p submission/notebooks
cp notebooks/demo_*_cat_a.py submission/notebooks/
cp notebooks/demo_*_cat_b.py submission/notebooks/
cp notebooks/training_*_cat_a.py submission/notebooks/
cp notebooks/training_*_cat_b.py submission/notebooks/
cp notebooks/evaluation.py submission/notebooks/

# Copy source code
