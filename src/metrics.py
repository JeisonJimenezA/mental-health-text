"""Primary regression metrics, clinically-derived metrics, and paired
comparison statistics used to decide between encoders.

Design note: regression is the only trained task. Severity-band and
screening-positive metrics are derived from the same continuous prediction
by applying fixed clinical cutoffs; they are not obtained from a separately
trained classifier. See PROTOCOL.md, section on task formulation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src import config
from src.splits import severity_band


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE and R^2. RMSE and R^2 always rank models identically on a
    fixed test set, since R^2 = 1 - MSE / Var(y_true) and Var(y_true) is
    constant across models; MAE is reported as an independent L1 check.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clip_to_valid_range(y_pred: np.ndarray, target: str) -> np.ndarray:
    lo, hi = config.SCORE_RANGE[target]
    return np.clip(y_pred, lo, hi)


def derive_severity_bands(y: np.ndarray) -> np.ndarray:
    return np.array([severity_band(s) for s in y])


def derive_screening_positive(y: np.ndarray, threshold: int = config.BINARY_THRESHOLD) -> np.ndarray:
    return (y >= threshold).astype(int)


def clinical_utility_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              threshold: int = config.BINARY_THRESHOLD) -> dict:
    """Severity-band and screening-positive metrics derived from a continuous
    prediction, for clinical interpretability alongside the regression result.
    """
    true_band, pred_band = derive_severity_bands(y_true), derive_severity_bands(y_pred)
    true_pos, pred_pos = derive_screening_positive(y_true, threshold), derive_screening_positive(y_pred, threshold)

    metrics = {
        "band_f1_macro": float(f1_score(true_band, pred_band, average="macro", zero_division=0)),
        "band_qwk": float(cohen_kappa_score(true_band, pred_band, weights="quadratic")),
        "band_confusion_matrix": confusion_matrix(true_band, pred_band, labels=[0, 1, 2, 3]).tolist(),
        "screening_f1": float(f1_score(true_pos, pred_pos, zero_division=0)),
        "screening_confusion_matrix": confusion_matrix(true_pos, pred_pos, labels=[0, 1]).tolist(),
    }
    if len(np.unique(true_pos)) > 1:
        metrics["screening_auc"] = float(roc_auc_score(true_pos, y_pred))
    return metrics


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
                         random_state: int = config.RANDOM_STATE,
                         confidence: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of paired per-participant differences."""
    rng = np.random.default_rng(random_state)
    n = len(diff)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(diff.mean()), float(lo), float(hi)


def paired_comparison(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Compare two encoders on the same held-out participants.

    Primary statistic: per-participant squared-error difference (backs the
    RMSE/R^2 comparison). Secondary: per-participant absolute-error
    difference (MAE robustness check). Both are tested with a paired
    bootstrap CI and a Wilcoxon signed-rank test.
    """
    se_a, se_b = (y_true - pred_a) ** 2, (y_true - pred_b) ** 2
    ae_a, ae_b = np.abs(y_true - pred_a), np.abs(y_true - pred_b)

    se_diff_mean, se_lo, se_hi = paired_bootstrap_ci(se_a - se_b)
    ae_diff_mean, ae_lo, ae_hi = paired_bootstrap_ci(ae_a - ae_b)

    se_p = 1.0 if np.allclose(se_a, se_b) else float(stats.wilcoxon(se_a, se_b).pvalue)
    ae_p = 1.0 if np.allclose(ae_a, ae_b) else float(stats.wilcoxon(ae_a, ae_b).pvalue)

    return {
        "squared_error_diff_mean": se_diff_mean,
        "squared_error_diff_ci95": (se_lo, se_hi),
        "squared_error_wilcoxon_p": se_p,
        "absolute_error_diff_mean": ae_diff_mean,
        "absolute_error_diff_ci95": (ae_lo, ae_hi),
        "absolute_error_wilcoxon_p": ae_p,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni correction over a family of pairwise comparisons.

    Returns, per comparison, its raw p-value, the rank-adjusted alpha it was
    tested against, and whether it survives correction.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    result = {}
    chain_holds = True
    for rank, (name, p) in enumerate(ordered):
        adjusted_alpha = alpha / (m - rank)
        significant = chain_holds and p <= adjusted_alpha
        chain_holds = chain_holds and significant
        result[name] = {"p_value": p, "adjusted_alpha": adjusted_alpha, "significant": significant}
    return result
