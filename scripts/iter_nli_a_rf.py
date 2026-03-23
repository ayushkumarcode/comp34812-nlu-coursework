"""NLI Cat A — Random Forest iteration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='nli'))
ext = NLIFeatureExtractor(use_spacy=True, use_glove=False)
ext.fit(train_df)
X_train, _ = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

model = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_split=5,
                                min_samples_leaf=2, class_weight='balanced',
                                random_state=42, n_jobs=1)
model.fit(X_tr, y_train)
y_pred = model.predict(X_dv)
metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A (RF)")
save_predictions(y_pred, PROJECT_ROOT / 'predictions' / 'nli_Group_34_A_rf.csv')
print("Done!")
