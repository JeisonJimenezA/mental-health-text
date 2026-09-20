"""Fixed train/test split and grouped cross-validation.

The hold-out split is fixed (seed=42) and reused unchanged across every
encoder arm so that all comparisons are paired on the same held-out
participants. This matches the split already used by the prior exploratory
notebooks (01_tfidf, 02_beto_embeddings, 021, 03_beto_finetune_hf), which is
kept unchanged to preserve continuity with that earlier work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src import config


def severity_band(score: float) -> int:
    """Standard PHQ-9 / GAD-7 severity band.

    0: minimal, 1: mild, 2: moderate, 3: moderately severe or severe.
    """
    s = int(round(score))
    if s < 5:
        return 0
    if s < 10:
        return 1
    if s < 15:
        return 2
    return 3


def stratify_key(df: pd.DataFrame) -> np.ndarray:
    """Combined PHQ9 x GAD7 severity-band key used to stratify both the
    hold-out split and the inner cross-validation folds."""
    phq9_band = df["PHQ9_Total"].apply(severity_band).to_numpy()
    gad7_band = df["GAD7_Total"].apply(severity_band).to_numpy()
    combined = phq9_band * 4 + gad7_band
    counts = np.bincount(combined, minlength=16)
    # collapse combinations with fewer than 2 members so sklearn can stratify
    return np.where(counts[combined] < 2, 0, combined)


def train_test_participants(df: pd.DataFrame,
                             test_size: float = config.TEST_SIZE,
                             random_state: int = config.RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Split participant IDs into train/test, stratified by PHQ9 x GAD7 severity band."""
    strat = stratify_key(df)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )
    return df.index[train_idx].to_numpy(), df.index[test_idx].to_numpy()


def assert_no_leakage(test_participants,
                       extra_df: pd.DataFrame,
                       original_participant_col: str = "original_participant") -> None:
    """Raise if any augmented row traces back to a held-out test participant.

    A prior version of this check was missing in an earlier iteration and
    caused a real leakage incident (inflated CV metrics vs. hold-out); this
    guard must run before any augmented row enters training or CV.
    """
    leaked = set(extra_df[original_participant_col]) & set(test_participants)
    if leaked:
        raise AssertionError(f"Data leakage detected: augmented rows reference test participants {leaked}")


def cv_splitter(groups: np.ndarray | None = None,
                 n_splits: int = config.N_CV_SPLITS,
                 random_state: int = config.RANDOM_STATE):
    """StratifiedGroupKFold when groups are given (augmented data), else StratifiedKFold.

    Grouping by original participant keeps a paraphrase and its source in the
    same fold, preventing paraphrase-level leakage across CV folds.
    """
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
