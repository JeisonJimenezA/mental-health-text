"""Spanish affective norms (Stadthagen-Gonzalez et al.), for the E11/E12 arms.

Two sets of psycholinguistic norms, both collected by asking native Spanish
speakers to rate Spanish words. That is the property that matters here: a
translated resource carries the translator's errors into the feature, and a
measured audit of NRC EmoLex's Spanish column on this corpus found "que"
labelled positive+trust (from the English slang "wot") firing 2,625 times,
"con" labelled joy (from "feat"), and "cuando" labelled negative (from the
archaic "wen"). A rating given to a Spanish word by a Spanish speaker cannot
fail that way.

- Valence and arousal, 14,031 words, rated 1-9 (Behavior Research Methods,
  2017). Covers 82% of the content words in this study's training answers.
- Five discrete emotions -- happiness, disgust, anger, fear, sadness --
  10,491 words, rated 1-5 (Behavior Research Methods, 2018). Covers 52%.
  This file also repeats valence and arousal for its subset, and adds the
  dominant part of speech of each word.

Four properties of the published files that the loader has to handle, all of
them verified rather than assumed:

- they are encoded ISO-8859-1, not UTF-8, so reading them as UTF-8 corrupts
  every accented word silently;
- the valence file carries ten empty trailing columns and pads some header
  names with spaces ("%ValenceRaters ");
- `Few_Raters` marks 89 entries of the emotion file whose ratings rest on too
  few raters to use;
- `Dominant_POS` spells the tags out ("ADJECTIVE", "ADVERB") where spaCy uses
  the Universal Dependencies short forms. Comparing them without mapping
  agrees on 56.8% of tokens; with the mapping, on 92.2%.

Nothing here is redistributable: the files are supplementary material of
their papers, fetched by the user and not part of this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import config
from src.lexicon import fold

RESOURCES_DIR = config.EXPERIMENT_ROOT / "src" / "resources"
VALENCE_FILE = "13428_2015_700_MOESM1_ESM.csv"
EMOTION_FILE = "13428_2017_962_MOESM1_ESM.csv"

ENCODING = "latin-1"

EMOTIONS: tuple[str, ...] = ("happiness", "disgust", "anger", "fear", "sadness")

# Rating scales, as published. Used for the neutral fallback and checked on
# load, so a file in a different edition cannot be read as if it were this one.
VALENCE_RANGE = (1.0, 9.0)
AROUSAL_RANGE = (1.0, 9.0)
EMOTION_RANGE = (1.0, 5.0)

# Declared before any feature was computed, not tuned afterwards. The valence
# and arousal cuts sit one point either side of the 1-9 midpoint, which is
# where a rating stops being neutral.
VALENCE_LOW = 4.0
VALENCE_HIGH = 6.0
AROUSAL_HIGH = 6.0

# What a feature reports when a text has no covered word at all. For valence
# and arousal that is the scale midpoint: zero would sit far below the floor
# of 1.15 and would read as the most distressed answer in the corpus. For the
# emotions the scale starts at "absent", so absence is the honest neutral.
# Either way the coverage features say the value was not measured.
VALENCE_NEUTRAL = 5.0
AROUSAL_NEUTRAL = 5.0
EMOTION_NEUTRAL = 1.0

# The emotion file's part-of-speech labels, in spaCy's vocabulary.
POS_ALIASES = {"NOUN": "NOUN", "ADJECTIVE": "ADJ", "VERB": "VERB", "ADVERB": "ADV"}


@dataclass(frozen=True)
class AffectiveNorms:
    """Accent- and case-folded lookups, one per rated dimension."""

    valence: dict[str, float]
    arousal: dict[str, float]
    emotions: dict[str, dict[str, float]]
    dominant_pos: dict[str, str]

    def valence_of(self, *forms: str) -> float | None:
        return self._first(self.valence, forms)

    def arousal_of(self, *forms: str) -> float | None:
        return self._first(self.arousal, forms)

    def emotions_of(self, *forms: str) -> dict[str, float] | None:
        """All five ratings for the first form that is rated, or None."""
        for form in forms:
            if form is None:
                continue
            key = fold(form)
            if key in self.emotions["fear"]:
                return {emotion: self.emotions[emotion][key] for emotion in EMOTIONS}
        return None

    def pos_agrees(self, spacy_pos: str, *forms: str) -> bool:
        """Whether the token's part of speech matches the word's dominant one.

        Decided on the first form the file labels, so a lemma the norms know
        about settles it before the surface form is consulted. Words the file
        does not label are treated as agreeing: the guard exists to reject a
        reading the raters were not shown, not to reject everything the file
        happens to be silent about. On this corpus it discards 8% of matches.
        """
        for form in forms:
            if form is None:
                continue
            dominant = self.dominant_pos.get(fold(form))
            if dominant is not None:
                mapped = POS_ALIASES.get(dominant)
                return mapped is None or mapped == spacy_pos
        return True

    @staticmethod
    def _first(table: dict[str, float], forms) -> float | None:
        for form in forms:
            if form is None:
                continue
            value = table.get(fold(form))
            if value is not None:
                return value
        return None


def _check_range(series: pd.Series, bounds: tuple[float, float], name: str) -> None:
    low, high = bounds
    observed = (series.min(), series.max())
    if observed[0] < low or observed[1] > high:
        raise ValueError(
            f"{name} ratings fall outside the published {low}-{high} scale "
            f"(observed {observed[0]}-{observed[1]}). This is not the edition "
            f"src/affective_norms.py was written for.")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Affective norms not found at {path}. The Stadthagen-Gonzalez "
            f"supplementary files are not redistributable and are not kept in "
            f"this repository; download them and place them in {RESOURCES_DIR}.")
    return path


@lru_cache(maxsize=1)
def load_norms(resources_dir: Path = RESOURCES_DIR) -> AffectiveNorms:
    """Both files, folded and validated. Cached: the tables are read once per
    process and every answer looks up against the same dictionaries."""
    valence_path = _require(Path(resources_dir) / VALENCE_FILE)
    emotion_path = _require(Path(resources_dir) / EMOTION_FILE)

    va = pd.read_csv(valence_path, encoding=ENCODING)
    va.columns = [c.strip() for c in va.columns]
    missing = {"Word", "ValenceMean", "ArousalMean"} - set(va.columns)
    if missing:
        raise ValueError(f"{valence_path.name} is missing columns {sorted(missing)}")
    va = va.dropna(subset=["Word", "ValenceMean", "ArousalMean"])
    _check_range(va.ValenceMean, VALENCE_RANGE, "Valence")
    _check_range(va.ArousalMean, AROUSAL_RANGE, "Arousal")
    va = va.assign(key=va.Word.astype(str).map(fold))

    emo = pd.read_csv(emotion_path, encoding=ENCODING)
    emo.columns = [c.strip() for c in emo.columns]
    emotion_columns = {f"{e.capitalize()}_Mean" for e in EMOTIONS}
    missing = ({"Word", "Few_Raters", "Dominant_POS"} | emotion_columns) - set(emo.columns)
    if missing:
        raise ValueError(f"{emotion_path.name} is missing columns {sorted(missing)}")
    # Few_Raters is a flag column: a mark means the ratings rest on too few
    # people to use, and a blank means they do not.
    emo = emo[emo.Few_Raters.isna()].dropna(subset=["Word"])
    for emotion in EMOTIONS:
        _check_range(emo[f"{emotion.capitalize()}_Mean"], EMOTION_RANGE, emotion)
    emo = emo.assign(key=emo.Word.astype(str).map(fold))

    # A handful of entries collapse onto one form once accents are folded
    # ("esta"/"está"); averaging them is the only defensible resolution, since
    # nothing in the file says which reading a token in a transcript carries.
    return AffectiveNorms(
        valence=va.groupby("key").ValenceMean.mean().to_dict(),
        arousal=va.groupby("key").ArousalMean.mean().to_dict(),
        emotions={e: emo.groupby("key")[f"{e.capitalize()}_Mean"].mean().to_dict()
                  for e in EMOTIONS},
        dominant_pos=emo.groupby("key").Dominant_POS.first().dropna().to_dict(),
    )


def norms_available(resources_dir: Path = RESOURCES_DIR) -> bool:
    """Whether both files are present, for callers that degrade rather than fail."""
    return all((Path(resources_dir) / name).exists()
               for name in (VALENCE_FILE, EMOTION_FILE))
