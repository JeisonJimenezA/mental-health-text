"""Append-only structured logs of experiment runs.

Two logs, at two granularities:
- experiment_log.csv: one row per (encoder, augmentation, target, seed) with
  the selected model's test metrics, so aggregate analysis reads a single
  flat table instead of walking per-notebook result folders.
- model_selection_log.csv: one row per (encoder, augmentation, target, seed,
  candidate regressor), recording every model the search tried and its CV
  score, not only the one that won. Keeps the model-selection step auditable
  rather than reporting only the winner.

Alongside them, predictions/ keeps one file per run holding the individual
test predictions. The aggregate logs record a run's RMSE, which cannot say
whether two arms differ on the same participants or on different ones;
paired significance tests (PROTOCOL.md, Section 10) need the per-participant
errors, and those are unrecoverable once a run is over.

All three carry `config_json`, the TrainingConfig the row was produced under,
and `logged_at`, so a row is self-describing once the tunable settings change.

Writes are append-only, keeping the full history of what was run. Reads go
through load_runs/load_trials, which keep only the most recent row per key:
re-running a notebook cell appends a second set of rows rather than
overwriting, and analysis must not silently average a metric over both.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

RUN_FIELDS = [
    "encoder", "family", "granularity", "notebook",
    "augmentation", "target", "seed", "fold", "split",
    "rmse", "mae", "r2",
    "band_f1_macro", "band_qwk", "band_confusion_matrix",
    "screening_f1", "screening_auc", "screening_confusion_matrix",
    "selected_model", "selected_params", "encoder_model", "search",
    "config_json", "logged_at", "notes",
]

TRIAL_FIELDS = [
    "encoder", "family", "granularity", "notebook",
    "augmentation", "target", "seed", "model_name", "cv_rmse", "is_selected",
    "encoder_model", "search", "config_json", "logged_at",
]

# `search` is part of the key so a defaults run and a tuned run of the same
# arm coexist instead of the newer silently superseding the older.
RUN_KEY = ["encoder", "augmentation", "target", "seed", "search"]
TRIAL_KEY = RUN_KEY + ["model_name"]

RUN_LOG_FILENAME = "experiment_log.csv"
TRIAL_LOG_FILENAME = "model_selection_log.csv"
PREDICTIONS_DIRNAME = "predictions"

# Fields holding nested structures (confusion matrices) that must round-trip
# through CSV as JSON rather than Python's default str() representation.
_JSON_FIELDS = {"band_confusion_matrix", "screening_confusion_matrix", "selected_params"}

# Metrics are written rounded: a float's full 17 significant digits are noise
# for an RMSE and make the file unreadable. Six decimals is well beyond any
# difference this experiment can resolve at n=48.
FLOAT_PRECISION = 6


def _serialize(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out = {}
    for field in fields:
        value = row.get(field, "")
        if field in _JSON_FIELDS and value != "":
            # Search results carry numpy scalars; .item() keeps ints as ints
            # rather than letting them fall through to a quoted string.
            out[field] = json.dumps(
                value, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        elif isinstance(value, float):
            out[field] = round(value, FLOAT_PRECISION)
        else:
            out[field] = value
    return out


def _migrate_header(path: Path, fields: list[str]) -> None:
    """Rewrites an existing log whose header predates a new column.

    Appending wider rows under a narrower header would silently misalign
    every column, so a log written by an older schema is rewritten once with
    the current header and blanks in the columns it never recorded.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        existing = rows[0].keys() if rows else None
    if existing is not None and list(existing) == fields:
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header == fields:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append(row: dict[str, Any], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    if not is_new:
        _migrate_header(path, fields)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(_serialize(row, fields))


def _stamp(row: dict[str, Any], cfg: config.TrainingConfig | None) -> dict[str, Any]:
    stamped = dict(row)
    stamped.setdefault("logged_at", datetime.now().isoformat(timespec="seconds"))
    if cfg is not None:
        stamped.setdefault("config_json", json.dumps(cfg.as_dict(), sort_keys=True))
    return stamped


def _row_dir(row: dict[str, Any]) -> Path:
    """A row carries its own family, augmentation and granularity, so it
    always knows which log it belongs in."""
    return config.results_dir(row["augmentation"],
                               row.get("family", config.FAMILY_FROZEN),
                               row.get("granularity", config.GRANULARITY_PARTICIPANT))


def log_run(row: dict[str, Any], cfg=None) -> None:
    _append(_stamp(row, cfg), RUN_FIELDS, _row_dir(row) / RUN_LOG_FILENAME)


def log_trials(rows: list[dict[str, Any]], cfg=None) -> None:
    for row in rows:
        _append(_stamp(row, cfg), TRIAL_FIELDS, _row_dir(row) / TRIAL_LOG_FILENAME)


def _prediction_path(row: dict[str, Any]) -> Path:
    """One file per run, named by the same key that identifies its log row."""
    stem = "__".join(str(row[field]) for field in RUN_KEY if field != "augmentation")
    return _row_dir(row) / PREDICTIONS_DIRNAME / f"{_safe(stem)}.csv"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def log_predictions(row: dict[str, Any], participants, y_true, y_pred) -> None:
    """Writes one run's per-participant test predictions.

    Overwrites rather than appends: unlike the metric logs, where keeping the
    history lets a re-run be compared against the original, a prediction file
    is a complete snapshot of one run and a second copy would only make the
    paired tests count some participants twice.
    """
    path = _prediction_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "participant": np.asarray(participants),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.round(np.asarray(y_pred, dtype=float), FLOAT_PRECISION),
    }).to_csv(path, index=False)


def load_predictions() -> pd.DataFrame:
    """Every persisted prediction, one row per (run, participant).

    Long rather than wide: arms do not all cover the same seeds, and a wide
    table would have to invent a fill value for the gaps. Pivot at the call
    site, where the intended pairing is known.
    """
    frames = []
    for path in sorted(config.RESULTS_DIR.rglob(f"{PREDICTIONS_DIRNAME}/*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        family, augmentation, granularity = path.parent.parent.parts[-3:]
        # Split from the right: an encoder name may itself contain the
        # separator, and splitting from the left would then misread every
        # field. The last three separators are the ones this format owns.
        encoder, target, seed, search = path.stem.rsplit("__", 3)
        frame = frame.assign(encoder=encoder, target=target, seed=int(seed),
                              search=search, family=family,
                              augmentation=augmentation, granularity=granularity)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_latest(path: Path, key: list[str]) -> pd.DataFrame:
    """Read a log, keeping only the most recently appended row per key."""
    df = pd.read_csv(path)
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def load_runs(augmentation: str, family: str = config.FAMILY_FROZEN,
               granularity: str = config.GRANULARITY_PARTICIPANT) -> pd.DataFrame:
    return _load_latest(
        config.results_dir(augmentation, family, granularity) / RUN_LOG_FILENAME, RUN_KEY)


def load_trials(augmentation: str, family: str = config.FAMILY_FROZEN,
                 granularity: str = config.GRANULARITY_PARTICIPANT) -> pd.DataFrame:
    return _load_latest(
        config.results_dir(augmentation, family, granularity) / TRIAL_LOG_FILENAME, TRIAL_KEY)


def load_all_runs() -> pd.DataFrame:
    """Every run ever logged, from every family, augmentation and
    granularity, in one frame. Cross-family comparisons are a groupby on
    the resulting columns rather than a manual concat at each call site.
    """
    frames = []
    for path in sorted(config.RESULTS_DIR.rglob(RUN_LOG_FILENAME)):
        frame = _load_latest(path, RUN_KEY)
        if frame.empty:
            continue
        # The directory identifies the run even for rows written before these
        # columns existed, so it fills anything the row left blank.
        from_path = dict(zip(("family", "augmentation", "granularity"), path.parent.parts[-3:]))
        for column, value in from_path.items():
            frame[column] = frame[column].fillna(value) if column in frame else value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_confusion_matrix(value: str):
    """Parse a confusion-matrix cell written by log_run back into a list of lists."""
    return json.loads(value)
