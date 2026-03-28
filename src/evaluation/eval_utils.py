"""
Evaluation utilities shared by both tracks. Confusion matrices,
McNemar's test, bootstrap CIs, error overlap analysis.
"""

import numpy as np
from sklearn import metrics as sklearn_metrics
from scipy import stats


def confusion_matrix_stats(y_true, y_pred):
    """Confusion matrix plus per-class P/R/F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = sklearn_metrics.classification_report(
        y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )

    return {
        'confusion_matrix': cm,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'class_0': report['0'],
        'class_1': report['1'],
        'accuracy': report['accuracy'],
    }


def mcnemars_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test with continuity correction. Returns chi2, p-value, etc."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = np.sum(correct_a & correct_b)
    n01 = np.sum(correct_a & ~correct_b)
    n10 = np.sum(~correct_a & correct_b)
    n11 = np.sum(~correct_a & ~correct_b)

    contingency = np.array([[n00, n01], [n10, n11]])

    b = float(n01)
    c = float(n10)

    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'contingency': contingency,
        'n_both_correct': int(n00),
        'n_only_a_correct': int(n01),
        'n_only_b_correct': int(n10),
        'n_both_wrong': int(n11),
        'chi2': chi2,
        'p_value': p_value,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000,
                                   ci=0.95, random_state=42):
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (default 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Dict with point estimate, CI lower, CI upper, and all bootstrap scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)

    n = len(y_true)
    point_estimate = metric_fn(y_true, y_pred)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return {
        'point_estimate': point_estimate,
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_level': ci,
        'n_bootstrap': n_bootstrap,
        'bootstrap_scores': bootstrap_scores,
    }


def bootstrap_macro_f1_ci(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Convenience function for macro F1 bootstrap CI."""
    def macro_f1(y_t, y_p):
        return sklearn_metrics.f1_score(y_t, y_p, average='macro', zero_division=0)
    return bootstrap_confidence_interval(y_true, y_pred, macro_f1,
                                          n_bootstrap=n_bootstrap,
                                          random_state=random_state)


def bootstrap_mcc_ci(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Convenience function for MCC bootstrap CI."""
    def mcc(y_t, y_p):
        return sklearn_metrics.matthews_corrcoef(y_t, y_p)
    return bootstrap_confidence_interval(y_true, y_pred, mcc,
                                          n_bootstrap=n_bootstrap,
                                          random_state=random_state)


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric_fn,
                           n_bootstrap=1000, random_state=42):
    """Test whether model B is significantly better than model A via paired bootstrap.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        metric_fn: Function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed.

    Returns:
        Dict with point estimates, mean difference, p-value, CI of difference.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    rng = np.random.RandomState(random_state)

    n = len(y_true)
    score_a = metric_fn(y_true, y_pred_a)
    score_b = metric_fn(y_true, y_pred_b)
    observed_diff = score_b - score_a

    diffs = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        diff = metric_fn(y_true[indices], y_pred_b[indices]) - \
               metric_fn(y_true[indices], y_pred_a[indices])
        diffs.append(diff)

    diffs = np.array(diffs)
    p_value = np.mean(diffs <= 0)  # proportion where B is not better

    return {
        'score_a': score_a,
        'score_b': score_b,
        'observed_diff': observed_diff,
        'mean_diff': np.mean(diffs),
        'p_value': p_value,
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5),
        'significant_005': p_value < 0.05,
    }


def cohens_kappa(y_pred_a, y_pred_b):
    """Compute Cohen's kappa between two sets of predictions.

    Args:
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Float kappa value.
    """
    return sklearn_metrics.cohen_kappa_score(
        np.asarray(y_pred_a), np.asarray(y_pred_b)
    )


def error_overlap_analysis(y_true, y_pred_a, y_pred_b, name_a='Model A', name_b='Model B'):
    """Analyze overlap in errors between two models.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        name_a: Display name for model A.
        name_b: Display name for model B.

    Returns:
        Dict with error counts and overlap statistics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    errors_a = set(np.where(y_pred_a != y_true)[0])
    errors_b = set(np.where(y_pred_b != y_true)[0])

    shared_errors = errors_a & errors_b
    only_a_errors = errors_a - errors_b
    only_b_errors = errors_b - errors_a

    return {
        f'{name_a}_errors': len(errors_a),
        f'{name_b}_errors': len(errors_b),
        'shared_errors': len(shared_errors),
        f'only_{name_a}_errors': len(only_a_errors),
        f'only_{name_b}_errors': len(only_b_errors),
        'shared_error_indices': sorted(shared_errors),
        f'only_{name_a}_error_indices': sorted(only_a_errors),
        f'only_{name_b}_error_indices': sorted(only_b_errors),
    }
