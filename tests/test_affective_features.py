"""Tests for the affective-norms loader and the E11/E12 feature block.

The failure modes here are the ones that do not raise. A resource read in the
wrong encoding still loads, with every accented word silently unmatched. A
neutral fallback of 0.0 on a 1-9 valence scale does not crash; it reports the
most distressed answer in the corpus for a text nothing was measured on. A
part-of-speech guard that never fires looks exactly like one that works.

Runs standalone (python tests/test_affective_features.py) or under pytest.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import affective_norms as an
from src import lexical_features as lf

AVAILABLE = an.norms_available()
NLP = lf.get_nlp()


def _affect(text: str) -> dict:
    return lf.extract_answer(text, NLP, name="affect")


# --- Loader --------------------------------------------------------------

def test_norms_load_and_are_not_empty():
    norms = an.load_norms()
    assert len(norms.valence) > 10_000, "valence table far smaller than published"
    assert len(norms.arousal) == len(norms.valence)
    assert set(norms.emotions) == set(an.EMOTIONS)
    assert len(norms.emotions["fear"]) > 9_000


def test_accented_words_survive_the_encoding():
    """The files are ISO-8859-1. Read as UTF-8 they still parse, but every
    accented word comes back mangled and silently stops matching.
    """
    norms = an.load_norms()
    for word in ["ansiedad", "corazon", "adiccion", "energia"]:
        assert word in norms.valence, f"{word} missing: the file was read as the wrong encoding"


def test_ratings_stay_inside_the_published_scales():
    norms = an.load_norms()
    assert all(an.VALENCE_RANGE[0] <= v <= an.VALENCE_RANGE[1] for v in norms.valence.values())
    assert all(an.AROUSAL_RANGE[0] <= v <= an.AROUSAL_RANGE[1] for v in norms.arousal.values())
    for emotion in an.EMOTIONS:
        values = norms.emotions[emotion].values()
        assert all(an.EMOTION_RANGE[0] <= v <= an.EMOTION_RANGE[1] for v in values)


def test_known_ratings_have_the_expected_direction():
    """A sanity check on the resource itself: if these came back reversed,
    every sign in the arm would be backwards and nothing would flag it.
    """
    norms = an.load_norms()
    assert norms.valence_of("triste") < 4.0
    assert norms.valence_of("feliz") > 6.0
    assert norms.valence_of("muerte") < norms.valence_of("amigo")
    assert norms.arousal_of("tranquilo") < norms.arousal_of("panico")


def test_pos_guard_uses_the_files_own_spelling():
    """Dominant_POS says ADJECTIVE and ADVERB where spaCy says ADJ and ADV.
    Without the alias map the guard rejects every adjective and adverb, which
    on this corpus is the difference between 57% and 92% agreement.
    """
    assert an.POS_ALIASES["ADJECTIVE"] == "ADJ"
    assert an.POS_ALIASES["ADVERB"] == "ADV"
    norms = an.load_norms()
    labelled = [w for w, p in norms.dominant_pos.items() if p == "ADJECTIVE"]
    assert labelled, "no word labelled ADJECTIVE; the column was not read"
    assert norms.pos_agrees("ADJ", labelled[0])
    assert not norms.pos_agrees("NOUN", labelled[0])


def test_unlabelled_words_do_not_trip_the_guard():
    norms = an.load_norms()
    assert norms.pos_agrees("NOUN", "palabrainventadaquenoexiste")


# --- Block ---------------------------------------------------------------

def test_block_returns_exactly_the_declared_names():
    produced = list(_affect("Me siento bastante cansada.").keys())
    assert produced == list(lf.answer_features("affect"))


def test_negative_text_scores_lower_valence_than_positive_text():
    negative = _affect("Fue horrible, un desastre espantoso y doloroso.")
    positive = _affect("Fue maravilloso, un exito precioso y divertido.")
    assert negative["val_mean"] < positive["val_mean"]
    assert negative["val_low_rate"] > positive["val_low_rate"]
    assert negative["val_min"] < positive["val_min"]


def test_extremes_keep_a_single_dark_word_a_mean_would_flatten():
    """The reason val_min and the emotion maxima are in the set: one extreme
    word among many neutral ones leaves the mean where it was.
    """
    neutral = "Estudio en la universidad, tomo el bus y llego a casa por la tarde."
    with_extreme = neutral + " A veces pienso en el suicidio."
    a, b = _affect(neutral), _affect(with_extreme)
    assert b["val_min"] < a["val_min"], "the minimum did not register the added word"
    assert abs(b["val_mean"] - a["val_mean"]) < abs(b["val_min"] - a["val_min"])


def test_neutral_fallback_is_the_scale_midpoint_not_zero():
    """A text with nothing rated must not read as maximally negative. The
    valence floor in the resource is 1.15, so 0.0 would sit below every real
    observation and rank that answer as the most distressed in the corpus.
    """
    empty = _affect("")
    assert empty["val_mean"] == an.VALENCE_NEUTRAL
    assert empty["val_min"] == an.VALENCE_NEUTRAL
    assert empty["aro_mean"] == an.AROUSAL_NEUTRAL
    assert empty["affect_coverage"] == 0.0
    assert empty["emo_coverage"] == 0.0
    for emotion in an.EMOTIONS:
        assert empty[f"emo_{emotion}_mean"] == an.EMOTION_NEUTRAL


def test_coverage_is_a_proportion_and_reports_what_was_measured():
    f = _affect("Estudio mucho y duermo poco ultimamente.")
    assert 0.0 <= f["affect_coverage"] <= 1.0
    assert 0.0 <= f["emo_coverage"] <= 1.0
    # valence covers about 82% of content words against 52% for the emotions,
    # so the two must not be collapsed into one control column.
    assert f["affect_coverage"] >= f["emo_coverage"]


def test_no_affect_feature_is_ever_nan_or_infinite():
    for text in ["", "   ", "Si.", "...", "[LUGAR]", "mmm", "No."]:
        for name, value in _affect(text).items():
            assert math.isfinite(value), f"{name} not finite on {text!r}"
    for name, value in lf.extract_participant([], NLP, name="affect").items():
        assert math.isfinite(value), f"{name} not finite with no answers"


def test_rates_and_means_are_invariant_to_repeating_the_text():
    """Doubling a text doubles the rated words and their denominators, so a
    mean, an extreme or a rate does not move. A count that escaped
    normalization does.

    val_p05 is exempt: a percentile interpolates between order statistics, so
    it shifts slightly when the sample doubles even though the distribution
    is identical. That is the estimator, not a normalization bug.
    """
    text = "Me preocupa mucho el examen y no logro descansar."
    once, twice = _affect(text), _affect(text + " " + text)
    for name in lf.answer_features("affect"):
        if name == "val_p05":
            continue
        assert abs(once[name] - twice[name]) < 1e-6, \
            f"{name} moved when the text was repeated: {once[name]} -> {twice[name]}"


# --- Feature sets --------------------------------------------------------

def test_the_three_sets_have_the_declared_widths():
    assert len(lf.answer_features("lexical")) == 51
    assert len(lf.participant_features("lexical")) == 60
    assert len(lf.answer_features("affect")) == 21
    assert len(lf.participant_features("affect")) == 25
    assert len(lf.answer_features("lexical-affect")) == 72
    assert len(lf.participant_features("lexical-affect")) == 85


def test_e10_is_unchanged_by_the_arrival_of_the_affect_block():
    """E10 is already logged under results/frozen/. Its columns, their order
    and its version string are what those rows were produced with, so adding
    a block must leave all three exactly as they were.
    """
    assert lf.feature_set("lexical").version == "lexical-v1"
    assert lf.ANSWER_FEATURES == lf.answer_features("lexical")
    assert lf.PARTICIPANT_FEATURES == lf.participant_features("lexical")
    assert "affect" not in lf.feature_set("lexical").blocks
    assert not any(name.startswith(("val_", "aro_", "emo_"))
                   for name in lf.PARTICIPANT_FEATURES)


def test_combined_set_is_the_union_in_a_fixed_order():
    combined = lf.answer_features("lexical-affect")
    assert combined[:51] == lf.answer_features("lexical")
    assert combined[51:] == lf.answer_features("affect")


def test_affect_set_carries_no_length_features():
    """E11 is defined by being about affect. n_words_total is not, and
    leaving it out is what makes the E11-E12 contrast mean something.
    """
    names = lf.participant_features("affect")
    assert not any(name in names for name in lf.STRUCTURAL_FEATURES)


def test_between_answer_spread_is_distinct_from_within_answer_spread():
    names = lf.participant_features("affect")
    assert "val_sd" in names        # dispersion inside one answer
    assert "val_mean_sd" in names   # movement between a person's answers


def test_unknown_feature_set_is_refused():
    for call in (lambda: lf.feature_set("nope"),
                 lambda: lf.answer_features("nope")):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("an unknown feature set was accepted")


# --- Over the real training corpus ---------------------------------------

def _train_matrix(name):
    from src import data, splits
    frame = data.load_dataset()
    train_ids, _ = splits.train_test_participants(frame)
    return lf.extract_frame(frame.loc[train_ids], name=name)


def test_no_affect_feature_is_constant_over_the_training_participants():
    """Catches a rating column that never resolves, a guard that rejects
    everything, and a threshold no answer ever crosses. Reads training
    participants only; nothing here touches the held-out set.
    """
    matrix = _train_matrix("affect")
    spread = matrix.std(axis=0)
    dead = sorted(spread[spread == 0].index)
    assert not dead, f"affect features with no variance over the training set: {dead}"


def test_training_matrix_is_finite_and_correctly_shaped():
    matrix = _train_matrix("affect")
    assert list(matrix.columns) == list(lf.participant_features("affect"))
    assert np.isfinite(matrix.to_numpy()).all()


def test_coverage_over_the_corpus_is_high_enough_to_mean_anything():
    """A resource can be valid and still not cover this corpus. Measured at
    82% of content words for valence and 52% for the emotions; the floors
    here are well below that, so this fails on a real regression rather than
    on ordinary variation.
    """
    matrix = _train_matrix("affect")
    assert matrix.affect_coverage.mean() > 0.60, "valence coverage collapsed"
    assert matrix.emo_coverage.mean() > 0.30, "emotion coverage collapsed"
    assert matrix.affect_coverage.min() > 0.0, "a participant matched nothing at all"


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if not AVAILABLE:
        print("Affective norms not present; nothing to test.")
        sys.exit(0)
    sys.exit(_run_all())
