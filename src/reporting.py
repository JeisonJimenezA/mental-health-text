"""Spreadsheet exports of the logged results.

The logs themselves stay CSV: they are appended to one row at a time during
a run, which a text file does safely, while rewriting a workbook on every
row would be slow and would lose everything if a run died mid-write. A CSV
also stays greppable and repairable by hand.

What gets exported for reading is a workbook. It carries typed numbers, so
the locale problem that plagued the CSV copies disappears: an Excel set to
Spanish no longer reads 5.5167 as five and a half trillion, because the cell
holds a number rather than text to be parsed.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src import config, logging_utils

METRICS = ["rmse", "mae", "r2"]
CLINICAL = ["band_f1_macro", "band_qwk", "screening_f1", "screening_auc"]
GROUP = ["family", "granularity", "encoder", "search", "target"]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across seeds for every arm.

    Flattened to single-level columns (rmse_mean, rmse_sd, ...) because a
    two-level header is awkward to read in a spreadsheet and awkward to paste
    into a document.
    """
    grouped = runs.groupby(GROUP)
    summary = grouped[METRICS].agg(["mean", "std"]).round(3)
    summary.columns = [f"{metric}_{'sd' if stat == 'std' else stat}"
                        for metric, stat in summary.columns]
    summary["seeds"] = grouped.size()
    return summary.sort_values("rmse_mean").reset_index()


def best_per_seed(runs: pd.DataFrame) -> pd.DataFrame:
    """Which arm won each seed, per target. Descriptive: choosing an arm by
    its test score is selection on the test set, so this reads stability,
    not a ranking to act on."""
    index = runs.groupby(["target", "seed"])["rmse"].idxmin()
    columns = ["target", "seed", "family", "granularity", "encoder",
                "selected_model", "search"] + METRICS
    return runs.loc[index, columns].sort_values(["target", "seed"]).reset_index(drop=True)


def _format_sheet(worksheet, frame: pd.DataFrame) -> None:
    """Freezes the header and widens columns to their content, so the sheet
    is readable without manual fiddling."""
    worksheet.freeze_panes = "A2"
    for position, column in enumerate(frame.columns, start=1):
        longest = max([len(str(column))] + [len(str(v)) for v in frame[column].head(200)])
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=position).column_letter].width = min(longest + 2, 44)


def export(augmentation: str = config.AUGMENTATION_NONE) -> Path:
    """Writes a dated workbook to results/exports/ and returns its path."""
    runs = logging_utils.load_all_runs()
    runs = runs[runs["augmentation"] == augmentation]
    if runs.empty:
        raise SystemExit(f"No runs logged for augmentation={augmentation!r}")

    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    workbook = config.EXPORTS_DIR / f"resultados_{augmentation}_{date.today():%Y%m%d}.xlsx"

    sheets = {
        "resumen": summarize(runs),
        "clinicas": runs.groupby(GROUP)[CLINICAL].mean().round(3).reset_index(),
        "mejor_por_semilla": best_per_seed(runs),
        # config_json is provenance for the logs, not something to read in a cell.
        "corridas": runs.drop(columns=["config_json"], errors="ignore"),
    }

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _format_sheet(writer.sheets[name], frame)
    return workbook
