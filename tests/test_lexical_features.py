"""Tests for the lexical arm's feature extractor.

Counting features fail silently. Nothing raises when a rate is computed over
the wrong denominator or a lexicon entry can never match: the column stays in
the matrix, full of zeros or of plausible-looking wrong numbers, and the
model trains on it regardless. A prior project in this line shipped two
features that were constant at zero for months (a hashtag counter looking for
a token the cleaner never wrote, an emoji counter running after the encoder
that stripped emoji) and a lexicon whose accented entries could never match
the accent-stripped text it was matched against.

Standardization then destroys the evidence: the arm's matrix goes through
StandardScaler, so a rate that is ten times too large is indistinguishable
downstream from the correct one. A wrong scale cannot be caught after this
point, only here.

Four kinds of test, in the order they catch things:

1. Per block, on hand-written Spanish where the expected count is known.
2. Structural: the declared names are exactly the keys produced, in order,
   and nothing the extractor emits is NaN or infinite.
3. Over the real training corpus: no feature is constant. This is the one
   that catches the whole "always zero" family at once.
4. Invariance: rates do not move when a text is repeated, which catches a
   raw count that escaped normalization.

Runs standalone (python tests/test_lexical_features.py) or under pytest.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import lexical_features as lf
from src import lexical_resources as lex
from src import lexicon

NLP = lf.get_nlp()


def _features(text: str) -> dict:
    return lf.extract_answer(text, NLP)


# --- 1. Per block --------------------------------------------------------

def test_person_counts_verb_inflection_not_just_pronouns():
    """The pro-drop case: no pronoun names the speaker, but three finite
    verbs are inflected for first person singular. A pronoun-list feature
    would score this answer at zero self-reference.
    """
    f = _features("Estoy cansada, no duermo bien y me siento agotada.")
    assert f["fps_verb_rate"] > 0, "first-person verbs not detected without a pronoun"
    assert f["fps_explicit_ratio"] == 0.0, "no nominative 'yo' in this sentence"


def test_person_counts_auxiliary_of_compound_tense():
    """In "he sentido" the person is marked on the auxiliary; the participle
    carries none. Counting only pos_=="VERB" would miss every compound tense.
    """
    f = _features("He sentido mucha ansiedad este semestre.")
    assert f["fps_verb_rate"] > 0, "person on AUX of a compound tense not counted"


def test_person_separates_explicit_yo_from_clitic_me():
    explicit = _features("Yo me he sentido mal.")
    implicit = _features("Me he sentido mal.")
    assert explicit["fps_explicit_ratio"] > 0
    assert implicit["fps_explicit_ratio"] == 0.0
    assert explicit["fps_pron_rate"] > implicit["fps_pron_rate"]


def test_person_distinguishes_first_second_third():
    f = _features("Yo estudio, tú trabajas y ellos descansan.")
    assert f["fps_verb_rate"] > 0 and f["sps_rate"] > 0 and f["tps_rate"] > 0
    assert 0.0 < f["self_other_ratio"] < 1.0


def test_negation_and_absolutist_terms():
    f = _features("Nunca me siento bien, no puedo con nada, todo es siempre igual.")
    assert f["negation_rate"] > 0
    assert f["absolutist_rate"] > 0
    assert f["absolutist_coverage"] > 0
    assert f["neg_per_sentence"] > 0

    neutral = _features("Ayer fui a la universidad y estudié con mis compañeros.")
    assert neutral["absolutist_rate"] == 0.0


def test_clinical_subscales_route_to_the_right_symptom():
    f = _features("No duermo, estoy agotada y me da mucha ansiedad.")
    assert f["clin_sueno"] > 0, "sleep terms not routed to the sleep subscale"
    assert f["clin_energia"] > 0, "exhaustion not routed to the energy subscale"
    assert f["clin_nervios"] > 0, "anxiety not routed to the nerves subscale"
    assert f["clin_apetito"] == 0.0, "appetite fired on a text with no appetite terms"
    assert f["clin_total_rate"] > 0


def test_clinical_matches_inflected_forms_through_the_lemma():
    """"estresada" and "dormí" are not lexicon entries; their lemmas are."""
    f = _features("Me siento estresada y anoche no dormí.")
    assert f["clin_estres"] > 0, "inflected 'estresada' not matched via lemma"
    assert f["clin_sueno"] > 0, "inflected 'dormí' not matched via lemma"


def test_clinical_matches_stem_changing_present_tense():
    """es_core_news_sm leaves "duermo" as its own lemma in some contexts, and
    "duermo" occurs 90 times in the training interviews. Without the
    inflection layer the sleep subscale misses the most common way anyone in
    this corpus reports a sleep problem.
    """
    assert _features("No duermo casi nada últimamente.")["clin_sueno"] > 0


def test_clinical_matches_feminine_adjectives():
    """Every adjective in CLINICAL_LEXICON is listed in the masculine, so a
    gap here under-counts women's answers specifically rather than at random.
    """
    for text, feature in [("Me siento muy cansada hoy.", "clin_energia"),
                          ("Ando bastante estresada.", "clin_estres"),
                          ("Estoy preocupada por el semestre.", "clin_preocupacion"),
                          ("Me siento sola.", "clin_soledad")]:
        assert _features(text)[feature] > 0, f"feminine form missed in {text!r}"


def test_clinical_homographs_are_guarded_by_part_of_speech():
    """"solo" is both "alone" and "only"; "como" is both "I eat" and "as".
    The written accent that used to separate them was abolished, so only the
    part of speech can, and the guard is deliberately biased towards
    precision: it counts a homograph only under the reading that is clinical.

    That trade is not free. es_core_news_sm tags the predicate adjective in
    "me siento solo casi siempre" as ADV, and sentence-initial "Como bien" as
    SCONJ, so both are missed. Recall on these two subscales is therefore
    below their true rate, while what they do count is right. The alternative
    is worse: "solo" meaning "only" is among the most frequent words in the
    corpus and would make clin_soledad a function-word counter. The
    diagnostic reports how often each subscale fires so this stays visible.
    """
    assert _features("Me siento solo.")["clin_soledad"] > 0
    assert _features("Siempre me he sentido solo.")["clin_soledad"] > 0
    assert _features("Me siento sola casi siempre.")["clin_soledad"] > 0

    assert _features("Solo quería contar eso, nada más.")["clin_soledad"] == 0.0, \
        "'solo' as 'only' counted as loneliness"
    assert _features("Me fue como siempre en los parciales.")["clin_apetito"] == 0.0, \
        "'como' as 'as/like' counted as appetite"


def test_morphosyntax_subjunctive_and_modals():
    subjunctive = _features("Ojalá pudiera dormir y que todo saliera bien.")
    assert subjunctive["mood_subjunctive_rate"] > 0

    indicative = _features("Ayer dormí ocho horas y desperté bien.")
    assert indicative["mood_subjunctive_rate"] == 0.0

    modal = _features("Tengo que estudiar y no puedo descansar.")
    assert modal["modal_rate"] > 0, "periphrastic 'tener que' not counted"


def test_morphosyntax_detects_past_and_present():
    """Only what es_core_news_sm actually resolves is asserted here.

    The model tags the past and the present reliably in running text, but it
    mis-tags the Spanish future simple often enough that an assertion on it
    would fail for the tagger's reasons rather than the extractor's:
    "comeré" comes back NOUN and "dormiré" ADJ in isolation. tense_fut_rate is
    kept as a feature because it does vary over the real corpus (the
    no-constant-feature test covers that), but it is the noisiest column in
    the matrix and the diagnostic reports it as such.
    """
    past = _features("El semestre pasado tuve muchos problemas y estuve muy estresado.")
    assert past["tense_past_rate"] > 0

    present = _features("Ahora me siento bien y duermo mucho mejor.")
    assert present["tense_pres_rate"] > 0
    assert present["tense_past_rate"] == 0.0


def test_syllable_counting():
    expected = {"casa": 2, "aéreo": 4, "día": 2, "bien": 1, "cansado": 3,
                "psicología": 5, "mmm": 1}
    for word, count in expected.items():
        assert lex.count_syllables(word) == count, f"{word}: expected {count}"


def test_richness_measures_are_length_robust():
    short = _features("Me siento bien, todo bien, bastante bien la verdad.")
    assert 0.0 < short["mattr_50"] <= 1.0
    assert short["mtld"] > 0
    assert 0.0 <= short["lexical_density"] <= 1.0


def test_mattr_on_a_text_shorter_than_the_window():
    assert lf._mattr(("a", "b", "a", "b")) == 0.5


def test_discourse_markers_and_hesitation():
    f = _features("Bueno, pues, o sea, es que... no sé, la verdad.")
    assert f["discourse_marker_rate"] > 0
    assert f["hesitation_rate"] > 0, "Whisper's ellipsis not counted as hesitation"

    clean = _features("Ayer terminé el trabajo final de la materia.")
    assert clean["hesitation_rate"] == 0.0


def test_discourse_repetition():
    f = _features("De vez en cuando tengo... Tengo mis altos y mis bajos.")
    assert f["repetition_rate"] > 0, "immediate self-repetition not counted"


def test_placeholders_are_stripped_before_parsing():
    """[LUGAR] and [NOMBRE] are review artifacts, not speech. Left in, they
    inflate the noun rate and the word count.
    """
    with_placeholder = _features("Vivo en [LUGAR] con mi familia.")
    without = _features("Vivo en con mi familia.")
    assert with_placeholder["pos_noun_rate"] == without["pos_noun_rate"]


# --- 2. Structural -------------------------------------------------------

def test_declared_names_match_produced_keys_exactly():
    produced = list(_features("Me siento un poco cansada, pero bien.").keys())
    assert produced == list(lf.ANSWER_FEATURES), "block outputs drifted from declarations"


def test_participant_names_match_produced_keys_exactly():
    produced = list(lf.extract_participant(
        ["Estoy bien.", "No he dormido mucho."], NLP).keys())
    assert produced == list(lf.PARTICIPANT_FEATURES)


def test_feature_names_by_granularity():
    from src import config
    assert lf.feature_names(config.GRANULARITY_QUESTION) == lf.ANSWER_FEATURES
    assert lf.feature_names(config.GRANULARITY_PARTICIPANT) == lf.PARTICIPANT_FEATURES


def test_no_feature_is_ever_nan_or_infinite():
    """StandardScaler and most of the regressors in regression_models.py do
    not accept missing values, so the extractor's contract is that it never
    emits one, including on the degenerate inputs.
    """
    for text in ["", "   ", "Sí.", "...", "[LUGAR]", "mmm", "No."]:
        for name, value in _features(text).items():
            assert math.isfinite(value), f"{name} not finite on {text!r}"
    for name, value in lf.extract_participant([], NLP).items():
        assert math.isfinite(value), f"{name} not finite for a participant with no answers"
    for name, value in lf.extract_participant(["Solo una respuesta."], NLP).items():
        assert math.isfinite(value), f"{name} not finite for a single-answer participant"


def test_blocks_do_not_collide_on_a_feature_name():
    assert len(lf.ANSWER_FEATURES) == len(set(lf.ANSWER_FEATURES))
    assert len(lf.PARTICIPANT_FEATURES) == len(set(lf.PARTICIPANT_FEATURES))


# --- Lexicon partition ---------------------------------------------------

# The set as it stood before it was partitioned into subscales. The vocabulary
# diagnostic under results/diagnostics/vocabulary/ and the masking of the
# adapted checkpoints under models/ were computed from exactly these terms, so
# the partition must reproduce them rather than quietly extend them.
_ORIGINAL_CLINICAL_LEXICON = frozenset({
    "interés", "placer", "decaído", "deprimido", "depresión", "esperanza", "desesperanza",
    "dormir", "sueño", "insomnio", "cansado", "cansancio", "energía", "agotado", "agotamiento",
    "apetito", "comer", "fracaso", "culpa", "culpable", "concentrar", "concentración",
    "lento", "inquieto", "inquietud", "muerte", "morir", "lastimar", "suicidio", "autolesión",
    "nervioso", "nervio", "ansioso", "ansiedad", "preocupar", "preocupación", "preocupado",
    "relajar", "irritable", "irritabilidad", "irritar", "miedo", "temor", "pánico", "terrible",
    "triste", "tristeza", "llorar", "estrés", "estresado", "estresar", "agobiado", "agobio",
    "abrumado", "angustia", "angustiado", "desmotivado", "motivación", "frustrado",
    "frustración", "solo", "soledad", "vacío", "autoestima", "sobrepensar", "bajón", "desánimo",
})


def test_subscales_reproduce_the_original_lexicon():
    assert lexicon.CLINICAL_LEXICON == _ORIGINAL_CLINICAL_LEXICON


def test_subscales_are_disjoint():
    seen = set()
    for name, terms in lexicon.CLINICAL_SUBSCALES.items():
        overlap = seen & terms
        assert not overlap, f"{name} repeats terms from another subscale: {overlap}"
        seen |= terms


def test_is_clinical_is_unaffected_by_the_inflection_layer():
    """The invariant that keeps results/diagnostics/vocabulary/ and the MLM
    masking of models/ reproducible: the inflections widen subscale_of only.
    """
    assert lexicon.is_clinical("dormir") is True
    assert lexicon.is_clinical("duermo") is False
    assert lexicon.subscale_of("duermo", "duermo", "VERB") == "sueno"


def test_enclitic_lemmas_resolve_to_their_verb():
    """spaCy returns "preocupar yo" for "preocuparme": the clitic becomes a
    second word of the lemma, and the full string matches no entry. Taking
    the first word generalizes to the reflexive form of every lexicon verb,
    including the ones CLINICAL_INFLECTIONS does not list.
    """
    for form, lemma, expected in [("preocuparme", "preocupar yo", "preocupacion"),
                                  ("concentrarme", "concentrar yo", "concentracion"),
                                  ("dormirme", "dormir yo", "sueno"),
                                  ("irritarme", "irritar yo", "irritabilidad")]:
        assert lexicon.subscale_of(form, lemma, "VERB") == expected, \
            f"enclitic lemma {lemma!r} did not resolve"


def test_inflections_point_at_declared_subscales():
    for form, subscale in lexicon.CLINICAL_INFLECTIONS.items():
        assert subscale in lexicon.CLINICAL_SUBSCALES, f"{form} -> unknown subscale {subscale}"


def test_inflections_do_not_contradict_the_base_lexicon():
    """A form listed both as a lexicon term and as an inflection of another
    symptom would resolve differently depending on which lookup ran first.
    """
    base = {lexicon.fold(term): name
            for name, terms in lexicon.CLINICAL_SUBSCALES.items() for term in terms}
    for form, subscale in lexicon.CLINICAL_INFLECTIONS.items():
        existing = base.get(lexicon.fold(form))
        assert existing in (None, subscale), \
            f"{form} is {existing} in the lexicon but {subscale} as an inflection"


def test_every_resource_entry_can_match_something():
    """A multi-word entry in a set matched against single-token lemmas can
    never fire. This is how "tener que" would have become a dead entry.
    """
    for name, terms in [("ABSOLUTIST", lex.ABSOLUTIST),
                        ("NEGATION_LEMMAS", lex.NEGATION_LEMMAS),
                        ("DISCOURSE_MARKERS", lex.DISCOURSE_MARKERS),
                        ("INTENSIFIERS", lex.INTENSIFIERS),
                        ("MODAL_LEMMAS", lex.MODAL_LEMMAS),
                        ("CLINICAL_LEXICON", lexicon.CLINICAL_LEXICON)]:
        multiword = [t for t in terms if " " in t]
        assert not multiword, f"{name} holds multi-word entries that cannot match: {multiword}"


# --- 3. Over the real training corpus ------------------------------------

def _train_matrix():
    from src import data, splits
    frame = data.load_dataset()
    train_ids, _ = splits.train_test_participants(frame)
    return lf.extract_frame(frame.loc[train_ids])


def test_no_feature_is_constant_over_the_training_participants():
    """The test that catches the whole silent-failure family.

    A feature that never varies carries no information and costs a column
    against 110 training rows. Whatever the cause -- a lexicon entry that can
    never match, a morphological tag the Spanish model does not emit, a
    denominator that cancels the numerator -- it shows up here as zero
    variance, and this is the only place it shows up at all.

    Reads training participants only. Nothing in this file touches the
    held-out set.
    """
    matrix = _train_matrix()
    spread = matrix.std(axis=0)
    dead = sorted(spread[spread == 0].index)
    assert not dead, f"features with no variance over the training set: {dead}"


def test_training_matrix_is_finite_and_correctly_shaped():
    matrix = _train_matrix()
    assert list(matrix.columns) == list(lf.PARTICIPANT_FEATURES)
    assert np.isfinite(matrix.to_numpy()).all(), "non-finite value in the training matrix"


# --- 4. Invariance -------------------------------------------------------

def test_rates_are_invariant_to_repeating_the_text():
    """Doubling a text doubles every count and every denominator, so a
    correctly normalized rate does not move. A count that escaped
    normalization does.

    Per-sentence and per-answer measures are exempt: repeating the text adds
    sentences, which is what they are meant to register.
    """
    text = "Me siento cansada y no duermo bien. Todo me preocupa mucho."
    once = _features(text)
    twice = _features(text + " " + text)

    exempt = {"mtld", "mattr_50", "hapax_rate", "absolutist_coverage", "clin_coverage",
              "fernandez_huerta", "mean_sentence_len", "sentence_len_sd",
              "neg_per_sentence", "repetition_rate"}
    for name in lf.ANSWER_FEATURES:
        if name in exempt:
            continue
        assert abs(once[name] - twice[name]) < 1e-6, \
            f"{name} changed when the text was repeated: {once[name]} -> {twice[name]}"


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
