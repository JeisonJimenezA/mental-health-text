"""Regression test for the leakage guard.

An earlier iteration of this project ran cross-validation without checking
that augmented rows never traced back to a held-out test participant, which
inflated CV metrics relative to the true hold-out performance. This test
locks in the fix. Runs standalone (python tests/test_splits.py) or under
pytest if installed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.splits import assert_no_leakage


def test_assert_no_leakage_passes_when_clean():
    extra = pd.DataFrame({"original_participant": ["UTB_001", "UTB_002"]})
    assert_no_leakage(test_participants=["UTB_050", "UTB_051"], extra_df=extra)


def test_assert_no_leakage_raises_on_overlap():
    extra = pd.DataFrame({"original_participant": ["UTB_001", "UTB_050"]})
    raised = False
    try:
        assert_no_leakage(test_participants=["UTB_050", "UTB_051"], extra_df=extra)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError on participant overlap"


if __name__ == "__main__":
    test_assert_no_leakage_passes_when_clean()
    test_assert_no_leakage_raises_on_overlap()
    print("OK")
