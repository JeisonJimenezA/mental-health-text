"""Loading of session transcriptions and PHQ-9 / GAD-7 total scores.

Mirrors the loading logic of the original 02_beto_embeddings notebook so the
participant set and label values are identical to the prior exploratory work.
"""
from pathlib import Path

import pandas as pd

from src import config


def load_transcriptions(data_dir: Path = config.DATA_DIR,
                         questions: list[str] = config.QUESTIONS) -> pd.DataFrame:
    """One row per participant, one column per question with transcription text.

    A question is set to None when its transcription file is missing or the
    session marked it as omitted ("OMITIDA"). Participants with no usable
    transcription at all are excluded.
    """
    records = []
    for session_dir in sorted(config.require_data_dir(data_dir).glob("UTB_*")):
        transcription_dir = session_dir / "features" / "transcription"
        if not transcription_dir.exists():
            continue
        row = {"participant": session_dir.name}
        has_any = False
        for question in questions:
            path = transcription_dir / f"audio_{question}_transcription.txt"
            text = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
            if text and text.upper() != "OMITIDA":
                row[question] = text
                has_any = True
            else:
                row[question] = None
        if has_any:
            records.append(row)
    return pd.DataFrame(records).set_index("participant")


def load_scores(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """One row per participant with PHQ9_Total and GAD7_Total."""
    records = []
    for xlsx_path in sorted(config.require_data_dir(data_dir)
                            .glob("UTB_*/datos_experimento_*.xlsx")):
        participant = xlsx_path.parent.name
        try:
            scores = pd.read_excel(xlsx_path, sheet_name="Puntajes_Totales")
            records.append({
                "participant": participant,
                "PHQ9_Total": float(scores["PHQ9_Total"].iloc[0]),
                "GAD7_Total": float(scores["GAD7_Total"].iloc[0]),
            })
        except Exception as exc:
            print(f"[WARN] Skipping {participant}: {exc}")
    return pd.DataFrame(records).set_index("participant")


def load_dataset(data_dir: Path = config.DATA_DIR,
                  questions: list[str] = config.QUESTIONS) -> pd.DataFrame:
    """Inner-join transcriptions and scores; drop participants missing either."""
    scores = load_scores(data_dir)
    transcriptions = load_transcriptions(data_dir, questions)
    df = scores.join(transcriptions, how="inner")
    return df.dropna(subset=["PHQ9_Total", "GAD7_Total"])


def load_augmented_dataset(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """LLM-paraphrase-augmented training rows: 110 training participants plus
    one paraphrase each, indexed by participant id. Every row (original and
    paraphrase) carries an `original_participant` column pointing back to the
    source training participant, for the CV group and the leakage guard
    (splits.assert_no_leakage) — this file only ever contains rows derived
    from training participants, never from the held-out test set.
    """
    path = data_dir / "augmented" / "train_augmented.csv"
    return pd.read_csv(path).set_index("participant")
