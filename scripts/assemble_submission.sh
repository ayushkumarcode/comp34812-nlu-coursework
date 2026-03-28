#!/bin/bash
# Assemble final submission zip for AV track (Cat A + Cat B)
# Usage: bash scripts/assemble_submission.sh

echo "Assembling AV track submission (Cat A + Cat B)"

cd /Users/kumar/Documents/University/Year3/NLU/project

rm -rf submission
mkdir -p submission

# Prediction files
mkdir -p submission/predictions
cp predictions/Group_34_A.csv submission/predictions/
cp predictions/Group_34_B.csv submission/predictions/

# Model cards
mkdir -p submission/model_cards
cp model_cards/model_card_sol1_cat_a.md submission/model_cards/
cp model_cards/model_card_sol2_cat_b.md submission/model_cards/

# Demo notebooks (.py and .ipynb)
mkdir -p submission/notebooks
cp notebooks/demo_av_cat_a.py submission/notebooks/
cp notebooks/demo_av_cat_a.ipynb submission/notebooks/
cp notebooks/demo_av_cat_b.py submission/notebooks/
cp notebooks/demo_av_cat_b.ipynb submission/notebooks/
cp notebooks/training_cat_a.py submission/notebooks/
cp notebooks/training_cat_a.ipynb submission/notebooks/
cp notebooks/training_cat_b.py submission/notebooks/
cp notebooks/training_cat_b.ipynb submission/notebooks/
cp notebooks/evaluation.py submission/notebooks/
cp notebooks/evaluation.ipynb submission/notebooks/

# Source code
cp -r src submission/
rm -rf submission/src/__pycache__ submission/src/**/__pycache__

# Remove NLI-specific dead code (we submitted AV track only)
rm -f submission/src/nli_feature_engineering.py
rm -f submission/src/nli_pipeline.py
rm -f submission/src/nli_spacy_features.py
rm -f submission/src/nli_tfidf_features.py
rm -f submission/src/models/nli_cat_b_model.py
rm -f submission/src/models/nli_cat_b_dataset.py
rm -f submission/src/training/train_nli_cat_b.py
rm -f submission/src/training/train_nli_ensemble.py
rm -f submission/src/training/run_nli_cat_a.py

# Models that are small enough (<10MB)
mkdir -p submission/models
# Cat B model is 3.1MB — include
cp models/av_cat_b_best.pt submission/models/ 2>/dev/null
# Cat A models (LightGBM ~1MB, scaler ~17KB, etc) — include
cp models/av_cat_a_lgbm.joblib submission/models/ 2>/dev/null
cp models/av_cat_a_scaler.joblib submission/models/ 2>/dev/null
cp models/av_cat_a_feature_names.joblib submission/models/ 2>/dev/null
cp models/av_cat_a_tfidf.joblib submission/models/ 2>/dev/null
cp models/av_cat_a_cosine.joblib submission/models/ 2>/dev/null

# README and poster
cp README.md submission/
cp poster/poster_av.pptx submission/poster.pptx 2>/dev/null || cp poster/poster.pptx submission/ 2>/dev/null

echo ""
echo "Contents:"
find submission -type f | sort

echo ""
echo "File sizes:"
du -sh submission/models/* 2>/dev/null

echo ""
echo "Creating zip..."
cd submission && zip -r ../Group_34_submission.zip . \
  -x '*.pyc' -x '__pycache__/*' -x '.DS_Store'
cd ..

echo ""
echo "Zip size:"
ls -lh Group_34_submission.zip
echo ""
echo "Done: Group_34_submission.zip"
