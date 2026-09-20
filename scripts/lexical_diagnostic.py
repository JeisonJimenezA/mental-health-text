"""Descriptive reading of the lexical feature set, before it becomes an arm.

Sixty features face 110 training participants. That ratio is only
defensible if the features carry something, so this measures what each one
does before any of them is evaluated against the held-out set: how often it
fires at all, how much it varies, how it ranks against PHQ-9 and GAD-7, and
how much of it is already said by another column.

Three readings:

- firing rate and spread. A feature that is zero for most participants has
  almost no support at n=110, whatever its correlation looks like. The
  symptom subscales are the ones at risk: a term list of three words fires
  only when someone uses one of those three words.
- rank correlation with each target, with a Benjamini-Hochberg adjustment
  over the whole set. Fifty-nine features against two targets is 118 tests,
  and at alpha 0.05 six of them are expected to look significant with no
  signal at all.
- collinearity. Several pairs overlap by construction (pos_pron_rate
  contains fps_pron_rate, mattr_50 and mtld measure the same thing,
  n_words_total and words_per_answer_mean differ only by the answer count).
  Ridge coefficients over collinear columns are not interpretable, and
  interpretability is the whole reason this arm exists.

Training participants only. Nothing here reads the held-out set: once the
arm has been scored on those 48 people, dropping a feature because of what
was seen there is selection on the test set, so the pruning decisions this
report supports have to be taken now.

Runs for one feature set at a time, so E10's lexical columns, E11's affective
ones and E12's union each get their own report rather than one table that
hides which arm a column belongs to.

Usage:
  ..\\env\\Scripts\\python.exe scripts\\lexical_diagnostic.py
  ..\\env\\Scripts\\python.exe scripts\\lexical_diagnostic.py --feature-set affect
  ..\\env\\Scripts\\python.exe scripts\\lexical_diagnostic.py --feature-set lexical-affect
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src import config, data, lexical_features, splits

OUTPUT_ROOT = config.RESULTS_DIR / "diagnostics" / "lexical"

# Pairs above this absolute Spearman correlation are reported as redundant.
COLLINEARITY_THRESHOLD = 0.80

# A feature firing for fewer than this share of participants has too little
# support to be read, independently of its correlation.
SPARSE_THRESHOLD = 0.20


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Adjusted p-values controlling the false discovery rate.

    Holm-Bonferroni is what PROTOCOL.md Section 10 prescribes for the
    confirmatory contrasts, where a false positive costs a claim. This is a
    descriptive screen over every feature at once, where the cost of a false
    positive is one column kept that should have been dropped, so the less
    conservative correction is the right one.
    """
    order = np.argsort(p_values)
    n = len(p_values)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * n / (rank + 1))
        adjusted[index] = running
    return adjusted


def feature_summary(matrix: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in matrix.columns:
        column = matrix[name]
        row = {
            "feature": name,
            "mean": column.mean(),
            "sd": column.std(),
            "min": column.min(),
            "max": column.max(),
            "pct_nonzero": 100.0 * (column != 0).mean(),
        }
        for target in ("PHQ9", "GAD7"):
            values = targets[config.TARGET_COLUMNS[target]]
            rho, p = stats.spearmanr(column, values)
            row[f"rho_{target}"] = 0.0 if np.isnan(rho) else rho
            row[f"p_{target}"] = 1.0 if np.isnan(p) else p
        rows.append(row)

    summary = pd.DataFrame(rows)
    for target in ("PHQ9", "GAD7"):
        summary[f"q_{target}"] = benjamini_hochberg(summary[f"p_{target}"].to_numpy())
    summary["abs_rho_max"] = summary[["rho_PHQ9", "rho_GAD7"]].abs().max(axis=1)
    return summary.sort_values("abs_rho_max", ascending=False).reset_index(drop=True)


def collinear_pairs(matrix: pd.DataFrame,
                    threshold: float = COLLINEARITY_THRESHOLD) -> pd.DataFrame:
    correlations = matrix.corr(method="spearman").abs()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    stacked = upper.stack()
    pairs = stacked[stacked >= threshold].sort_values(ascending=False)
    return pd.DataFrame({
        "feature_a": [a for a, _ in pairs.index],
        "feature_b": [b for _, b in pairs.index],
        "abs_spearman": pairs.to_numpy(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default=lexical_features.DEFAULT_FEATURE_SET,
                        choices=sorted(lexical_features.FEATURE_SETS),
                        help="which arm's columns to describe")
    args = parser.parse_args()
    spec = lexical_features.feature_set(args.feature_set)
    output_dir = OUTPUT_ROOT / args.feature_set

    frame = data.load_dataset()
    train_ids, _ = splits.train_test_participants(frame)
    train = frame.loc[train_ids]
    print(f"training participants: {len(train)}  (held-out set untouched)")

    print(f"extracting features ({args.feature_set}) ...", flush=True)
    matrix = lexical_features.extract_frame(train, name=args.feature_set)
    print(f"feature matrix: {matrix.shape[0]} x {matrix.shape[1]} ({spec.version})")

    summary = feature_summary(matrix, train)
    pairs = collinear_pairs(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
    pairs.round(4).to_csv(output_dir / "collinearity.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    columns = ["feature", "mean", "sd", "pct_nonzero", "rho_PHQ9", "q_PHQ9",
               "rho_GAD7", "q_GAD7"]

    print("\n=== strongest rank correlations with either target ===")
    print(summary[columns].head(20).round(3).to_string(index=False))

    survivors = summary[(summary.q_PHQ9 <= 0.05) | (summary.q_GAD7 <= 0.05)]
    print(f"\n=== features surviving Benjamini-Hochberg at q<=0.05: {len(survivors)} "
          f"of {len(summary)} ===")
    print(survivors[columns].round(3).to_string(index=False) if len(survivors)
          else "  none")

    sparse = summary[summary.pct_nonzero < 100 * SPARSE_THRESHOLD]
    print(f"\n=== features firing for fewer than {SPARSE_THRESHOLD:.0%} of "
          f"participants: {len(sparse)} ===")
    print(sparse[["feature", "pct_nonzero", "rho_PHQ9", "rho_GAD7"]]
          .round(3).to_string(index=False) if len(sparse) else "  none")

    print(f"\n=== collinear pairs (|rho| >= {COLLINEARITY_THRESHOLD}): {len(pairs)} ===")
    print(pairs.head(25).round(3).to_string(index=False) if len(pairs) else "  none")

    print(f"\nsaved to {output_dir}")


if __name__ == "__main__":
    main()
