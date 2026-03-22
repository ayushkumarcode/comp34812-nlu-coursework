"""
NLI Category A — Stacking Ensemble Training.
Base classifiers: XGBoost, LightGBM, SVM-RBF, Logistic Regression.
Meta-learner: Logistic Regression.
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.scorer import compute_all_metrics, print_metrics


RANDOM_STATE = 42


def build_stacking_ensemble():
    """Build NLI stacking ensemble classifier.

    Returns:
        StackingClassifier with XGBoost, LightGBM, SVM, LR base + LR meta.
    """
    base_classifiers = [
        ('xgb', XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=5,
            gamma=0.1,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1,
            verbose=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ('svm', SVC(
            C=10,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=RANDOM_STATE,
        )),
        ('lr', LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ]

    meta_classifier = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    ensemble = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    return ensemble


def train_ensemble(X_train, y_train, X_dev=None, y_dev=None):
    """Train the stacking ensemble.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_dev: Optional dev feature matrix.
        y_dev: Optional dev labels.

    Returns:
        Tuple of (fitted_scaler, fitted_ensemble, dev_metrics).
    """
    print(f"Training data shape: {X_train.shape}")
    print(f"Label distribution: {np.bincount(y_train)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("Feature scaling done.")

    # Build and train ensemble
    ensemble = build_stacking_ensemble()
    print("Training stacking ensemble...")
    ensemble.fit(X_train_scaled, y_train)
    print("Ensemble training complete.")

    # Cross-validation
    cv_scores = cross_val_score(
        build_stacking_ensemble(), X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro',
        n_jobs=-1,
    )
    print(f"5-fold CV macro_f1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Evaluate on dev
    dev_metrics = None
    if X_dev is not None and y_dev is not None:
        X_dev_scaled = scaler.transform(X_dev)
        y_pred = ensemble.predict(X_dev_scaled)
        dev_metrics = compute_all_metrics(y_dev, y_pred)
        print_metrics(dev_metrics, "NLI Cat A — Dev Set Results")

    return scaler, ensemble, dev_metrics


def save_ensemble(scaler, ensemble, feature_extractor, save_dir='models'):
    """Save trained NLI ensemble and preprocessing objects."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    joblib.dump(scaler, save_dir / 'nli_cat_a_scaler.joblib')
    joblib.dump(ensemble, save_dir / 'nli_cat_a_ensemble.joblib')
    joblib.dump(feature_extractor.tfidf, save_dir / 'nli_cat_a_tfidf.joblib')
    if feature_extractor.feature_names:
        joblib.dump(feature_extractor.feature_names, save_dir / 'nli_cat_a_feature_names.joblib')
    print(f"NLI models saved to {save_dir}/")


def predict(X, scaler, ensemble):
    """Generate predictions."""
    X_scaled = scaler.transform(X)
    return ensemble.predict(X_scaled)


def predict_proba(X, scaler, ensemble):
    """Generate probability predictions."""
    X_scaled = scaler.transform(X)
    return ensemble.predict_proba(X_scaled)[:, 1]
