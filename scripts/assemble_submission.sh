#!/bin/bash
# Assemble final submission zip
# Usage: bash scripts/assemble_submission.sh [av|nli]

echo "Assembling NLI track submission (Cat A + Cat C)"

rm -rf submission
mkdir -p submission

# Prediction files
cp predictions/Group_34_A.csv submission/
cp predictions/Group_34_C.csv submission/

# Model cards
cp model_cards/model_card_nli_cat_a.md submission/
cp model_cards/model_card_nli_cat_c.md submission/

# Notebooks (NLI demos, training, evaluation)
mkdir -p submission/notebooks
cp notebooks/demo_nli_cat_a.py submission/notebooks/
cp notebooks/demo_nli_cat_c.py submission/notebooks/
cp notebooks/training_nli_cat_a.py submission/notebooks/
cp notebooks/training_nli_cat_c.py submission/notebooks/
cp notebooks/evaluation.py submission/notebooks/

# Copy source code
cp -r src submission/

# Copy README and poster
cp README.md submission/
cp poster/poster.pptx submission/ 2>/dev/null

echo "Contents:"
find submission -type f | sort

echo "Creating zip..."
cd submission && zip -r ../Group_34_submission.zip . \
  -x '*.pyc' -x '__pycache__/*'
echo "Done: Group_34_submission.zip"
