"""Unlabeled text from outside the project, for TSDAE adaptation only.

MentalRiskES 2026: written chat sessions in which people talk about their
anxiety and mood with a therapist. Close to the interviews in content, but a
different register (typed chat, Peninsular Spanish) and different people, so
it can enlarge the adaptation corpus without touching the held-out
participants. It never enters the downstream regression.

Only the patients' own messages are used:

- session_store.json holds every session's turns. The per-round files under
  rounds/ and gold-label/*/data repeat the same turns (verified identical),
  so reading them too would only duplicate the corpus.
- The therapist responses (option_1..3 in the task-2 files) are LLM output.
  They carry the generator's register rather than the patients', the same
  reason the interview paraphrases are excluded (PROTOCOL.md, Section 9),
  and at ~72k words they would outweigh the patient text several times over.
- The questionnaire answers and the competition predictions are labels and
  outputs, not text, and have no place in an unsupervised corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config


def load_mentalriskes_sessions(data_dir: Path = config.MENTALRISKES_DIR) -> dict[str, list[str]]:
    """{session id: patient messages in round order}, with whitespace
    collapsed and exact duplicates removed across the whole corpus (a repeated
    message is kept where it first appears)."""
    path = Path(data_dir) / "session_store.json"
    if not path.exists():
        raise FileNotFoundError(f"MentalRiskES session store not found at {path}")
    store = json.loads(path.read_text(encoding="utf-8"))

    sessions, seen = {}, set()
    for session_id in sorted(store):
        turns = []
        for turn in sorted(store[session_id]["rounds"], key=lambda r: r["round"]):
            text = " ".join(str(turn.get("patient_input") or "").split())
            if text and text not in seen:
                seen.add(text)
                turns.append(text)
        sessions[session_id] = turns
    return sessions


def load_mentalriskes_texts(data_dir: Path = config.MENTALRISKES_DIR) -> list[str]:
    """Patient messages, one per turn, in session and round order."""
    return [text for turns in load_mentalriskes_sessions(data_dir).values() for text in turns]
