"""
AV Category A — Stacking Ensemble Training.
Base classifiers: SVM-RBF, Random Forest, XGBoost.
Meta-learner: Logistic Regression.
"""

import numpy as np
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.scorer import compute_all_metrics, print_metrics, quick_score


# Reproducibility
RANDOM_STATE = 42


def build_stacking_ensemble():
    """Build the stacking ensemble classifier.

    Returns:
        StackingClassifier with SVM-RBF, Random Forest, XGBoost base + LR meta.
    """
    base_classifiers = [
        ('svm', SVC(
            C=10,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=RANDOM_STATE,
        )),
        ('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ('xgb', XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
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
        X_dev: Optional dev feature matrix for evaluation.
        y_dev: Optional dev labels.

    Returns:
        Tuple of (fitted_scaler, fitted_ensemble, dev_metrics).
    """
    print(f"Training data shape: {X_train.shape}")
    print(f"Training label distribution: {np.bincount(y_train)}")

    # Scale features (required for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("Feature scaling done.")

    # Build and train ensemble
    ensemble = build_stacking_ensemble()
    print("Training stacking ensemble (this may take a while)...")
    ensemble.fit(X_train_scaled, y_train)
    print("Ensemble training complete.")

    # Cross-validation score on training data
    cv_scores = cross_val_score(
        build_stacking_ensemble(), X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro',
        n_jobs=-1,
    )
    print(f"5-fold CV macro_f1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Evaluate on dev set if provided
    dev_metrics = None
    if X_dev is not None and y_dev is not None:
        X_dev_scaled = scaler.transform(X_dev)
        y_pred = ensemble.predict(X_dev_scaled)
        dev_metrics = compute_all_metrics(y_dev, y_pred)
        print_metrics(dev_metrics, "AV Cat A — Dev Set Results")

    return scaler, ensemble, dev_metrics


def save_ensemble(scaler, ensemble, feature_extractor, save_dir='models'):
    """Save trained ensemble and preprocessing objects.

    Args:
        scaler: Fitted StandardScaler.
        ensemble: Fitted StackingClassifier.
        feature_extractor: Fitted AVFeatureExtractor (contains TF-IDF, SVD).
        save_dir: Directory to save models.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    joblib.dump(scaler, save_dir / 'cat_a_scaler.joblib')
    joblib.dump(ensemble, save_dir / 'cat_a_ensemble.joblib')
    joblib.dump(feature_extractor.tfidf, save_dir / 'cat_a_tfidf.joblib')
    joblib.dump(feature_extractor.cosine, save_dir / 'cat_a_cosine.joblib')
    if feature_extractor.feature_names:
        joblib.dump(feature_extractor.feature_names, save_dir / 'cat_a_feature_names.joblib')
    print(f"Models saved to {save_dir}/")


def load_ensemble(save_dir='models'):
    """Load trained ensemble and preprocessing objects.

    Returns:
        Tuple of (scaler, ensemble, tfidf, cosine, feature_names).
    """
    save_dir = Path(save_dir)
    scaler = joblib.load(save_dir / 'cat_a_scaler.joblib')
    ensemble = joblib.load(save_dir / 'cat_a_ensemble.joblib')
    tfidf = joblib.load(save_dir / 'cat_a_tfidf.joblib')
    cosine = joblib.load(save_dir / 'cat_a_cosine.joblib')
    feature_names = joblib.load(save_dir / 'cat_a_feature_names.joblib')
    return scaler, ensemble, tfidf, cosine, feature_names


def predict(X, scaler, ensemble):
    """Generate predictions from feature matrix.

    Args:
        X: Feature matrix.
        scaler: Fitted StandardScaler.
        ensemble: Fitted StackingClassifier.

    Returns:
        Array of predictions (0/1).
    """
    X_scaled = scaler.transform(X)
    return ensemble.predict(X_scaled)


def predict_proba(X, scaler, ensemble):
    """Generate probability predictions.

    Args:
        X: Feature matrix.
        scaler: Fitted StandardScaler.
        ensemble: Fitted StackingClassifier.

    Returns:
        Array of probabilities for class 1.
    """
    X_scaled = scaler.transform(X)
    return ensemble.predict_proba(X_scaled)[:, 1]
