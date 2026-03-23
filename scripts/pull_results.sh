#!/bin/bash
# Pull trained models and predictions from CSF3
# Usage: bash scripts/pull_results.sh

echo "Pulling results from CSF3..."

# Create local dirs
mkdir -p models predictions

# Pull models
echo "Pulling models..."
scp csf3:~/scratch/nlu-project/models/*.pt models/ 2>/dev/null
scp csf3:~/scratch/nlu-project/models/*.joblib models/ 2>/dev/null

# Pull predictions
echo "Pulling predictions..."
scp csf3:~/scratch/nlu-project/predictions/*.csv predictions/ 2>/dev/null

# Pull logs
echo "Pulling logs..."
mkdir -p logs
scp csf3:~/scratch/nlu-project/logs/*.log logs/ 2>/dev/null

echo "Done. Contents:"
echo "--- Models ---"
ls -la models/
echo "--- Predictions ---"
ls -la predictions/
echo "--- Logs ---"
ls -la logs/
