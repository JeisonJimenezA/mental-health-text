"""Lexical features of a transcribed spoken answer (arm E10).

The frozen encoder arms turn an answer into a few hundred dimensions that
have no names. This one turns it into a few dozen that do: how often the
speaker refers to themselves, how much of what they say is negated, which
PHQ-9 and GAD-7 symptoms they name, how varied their vocabulary is. The
point is not to beat the embeddings — TF-IDF reaches R^2 = 0.015 on this
data, so the expectation for surface lexical signal is low — but to have one
arm whose coefficients can be read, and a floor the dense encoders can be
measured against.

The input is Whisper's transcription of a student answering an interview
question aloud, and two properties of that input shape everything here.

Length varies by a factor of sixty across participants (76 to 4754 words),
so every count is expressed as a rate per 100 words and verbosity enters
only through the four features that measure it deliberately. A raw count
would make most of this matrix a length proxy.

Whisper large-v3 normalizes disfluency away: the corpus holds three
occurrences of "eh" across 544 answers. Filler-sound features would be
constant at zero, so there are none. What survives of the spoken register is
the discourse markers ("o sea", "pues", "es que") and the ellipsis Whisper
writes where a speaker trails off, and those are what block G measures.

Person, number, tense, mood and negation come from spaCy's morphology rather
than from word lists. Spanish is pro-drop, so "estoy cansada" carries a
first person that no pronoun marks, and counting pronouns would undercount
it by however much each speaker happens to drop. The marking sits on the
finite verb — and, for compound tenses, on the auxiliary rather than the
participle ("he sentido" marks person on "he"), which is why the count is
over VerbForm=Fin regardless of coarse POS.

Features are declared in blocks, each of which names its own outputs up
front. Adding a family later (an affective lexicon, for instance) is
registering one more block and raising FEATURE_SET_VERSION; the column order
is derived from the registry, so nothing downstream has to be kept in sync
by hand. The version string is what the embedding cache stores as its
signature, so a feature added after a run invalidates that run's cached
matrix instead of being silently mixed into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import spacy

from src import affective_norms, config, lexical_resources as lex, lexicon

# Bumped whenever the feature set changes. Used as the cache signature.
FEATURE_SET_VERSION = "lexical-v1"

# Aggregated across a participant's answers as a standard deviation as well as
# a mean. Kept to five, because a spread column for every feature would double
# the width of a matrix that already faces only 110 training participants; the
# five chosen are the ones where within-person variation is itself plausible
# signal (does the self-focus, the negation, the symptom talk, the vocabulary
# or the sentence length change from question to question).
SD_FEATURES = ("fps_verb_rate", "negation_rate", "clin_total_rate",
               "mattr_50", "mean_sentence_len")

# The same idea for the affective arm: how much a person's tone, agitation,
# fear and sadness move between one question and the next. Emotional
# variability is a construct of its own in this literature, not a by-product
# of the means, which is why the affective arm carries it even though it is
# the least established of its four groups.
AFFECT_SD_FEATURES = ("val_mean", "aro_mean", "emo_fear_mean", "emo_sadness_mean")

MATTR_WINDOW = 50
MTLD_THRESHOLD = 0.72

# POS tags counted as content-bearing for lexical density.
_CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})

# What the affective norms are looked up for. Narrower than _CONTENT_POS,
# which includes proper nouns: a name carries no rating anyone gave it.
_AFFECT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})


# --- spaCy ---------------------------------------------------------------

_nlp = None


def get_nlp():
    """The Spanish pipeline, with the entity recognizer switched off.

    The parser stays on, unlike in encoders.py: sentence boundaries in
    spaCy's Spanish models come from the parser, and three of these features
    are per-sentence. The morphologizer and lemmatizer are what the rest
    reads, and both are on by default.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_sm", disable=["ner"])
    return _nlp


# --- Block registry ------------------------------------------------------

@dataclass(frozen=True)
class Block:
    name: str
    features: tuple[str, ...]
    fn: Callable


BLOCKS: list[Block] = []


def _block(name: str, features: list[str]):
    """Registers one feature family. The declared names are the contract: a
    block returning a different set of keys is a bug the tests catch, not
    something the matrix silently absorbs.
    """
    def decorate(fn):
        BLOCKS.append(Block(name, tuple(features), fn))
        return fn
    return decorate


@dataclass(frozen=True)
class Context:
    """What every block needs about one answer, computed once."""
    text: str
    n_words: int          # alphabetic tokens only
    n_sentences: int
    sentence_lengths: tuple[int, ...]
    lemmas: tuple[str, ...]   # folded lemmas of the alphabetic tokens

    def rate(self, count: float) -> float:
        """Per 100 alphabetic words. Zero, not NaN, on an empty answer: the
        matrix goes to StandardScaler and then to regressors that do not
        accept missing values.
        """
        return 100.0 * count / self.n_words if self.n_words else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    """Proportion with an explicit zero fallback, so no block can emit NaN."""
    return numerator / denominator if denominator else 0.0


def _context(doc) -> Context:
    sentences = [s for s in doc.sents]
    lengths = tuple(sum(1 for t in s if t.is_alpha) for s in sentences)
    lemmas = tuple(lexicon.fold(t.lemma_) for t in doc if t.is_alpha)
    return Context(text=doc.text, n_words=len(lemmas),
                   n_sentences=max(len(sentences), 1), sentence_lengths=lengths,
                   lemmas=lemmas)


def _person_of(token) -> str | None:
    """The token's grammatical person, for finite verbs and pronouns only.

    Restricting it to these two keeps each predicate counted once: in "he
    sentido" the auxiliary is finite and the participle is not, so the
    predicate contributes one first person rather than two.
    """
    if token.pos_ == "PRON" or token.morph.get("VerbForm") == ["Fin"]:
        person = token.morph.get("Person")
        return person[0] if person else None
    return None


def _is_singular(token) -> bool:
    return token.morph.get("Number") == ["Sing"]


# --- A: person -----------------------------------------------------------

@_block("person", ["fps_verb_rate", "fps_pron_rate", "fps_explicit_ratio",
                   "fpp_rate", "sps_rate", "tps_rate", "self_other_ratio"])
def _person_block(doc, ctx: Context) -> dict[str, float]:
    fps_verb = fps_pron = fps_nominative = fpp = sps = tps = 0
    for token in doc:
        person = _person_of(token)
        if person is None:
            continue
        if person == "1" and _is_singular(token):
            if token.pos_ == "PRON":
                fps_pron += 1
                # Case=Nom is what separates the stressed subject pronoun
                # "yo" from the clitic "me"; both are Person=1|Number=Sing.
                if token.morph.get("Case") == ["Nom"]:
                    fps_nominative += 1
            else:
                fps_verb += 1
        elif person == "1":
            fpp += 1
        elif person == "2":
            sps += 1
        elif person == "3":
            tps += 1

    first_person = fps_verb + fps_pron
    return {
        "fps_verb_rate": ctx.rate(fps_verb),
        "fps_pron_rate": ctx.rate(fps_pron),
        # Spanish drops the subject pronoun by default, so an explicit "yo" is
        # marked: this is the share of first-person predicates the speaker
        # chose to foreground themselves in, not a count of self-reference.
        "fps_explicit_ratio": _ratio(fps_nominative, fps_verb),
        "fpp_rate": ctx.rate(fpp),
        "sps_rate": ctx.rate(sps),
        "tps_rate": ctx.rate(tps),
        "self_other_ratio": _ratio(first_person, first_person + sps + tps),
    }


# --- B: negation and absolutism ------------------------------------------

@_block("negation", ["negation_rate", "absolutist_rate", "absolutist_coverage",
                     "neg_per_sentence"])
def _negation_block(doc, ctx: Context) -> dict[str, float]:
    negations = 0
    absolutist = 0
    absolutist_seen: set[str] = set()
    for token in doc:
        if not token.is_alpha:
            continue
        if token.morph.get("Polarity") == ["Neg"] or lex.is_negation(token.text, token.lemma_):
            negations += 1
        if lex.is_absolutist(token.text, token.lemma_):
            absolutist += 1
            absolutist_seen.add(lexicon.fold(token.lemma_))
    return {
        "negation_rate": ctx.rate(negations),
        "absolutist_rate": ctx.rate(absolutist),
        # Share of the absolutist vocabulary the speaker reaches for at all.
        # Type counts grow with length by construction, so this is expected to
        # correlate with verbosity; the diagnostic reports that correlation and
        # the feature is dropped if it turns out to measure nothing else.
        "absolutist_coverage": _ratio(len(absolutist_seen), len(lex.ABSOLUTIST)),
        "neg_per_sentence": _ratio(negations, ctx.n_sentences),
    }


# --- D: clinical subscales -----------------------------------------------

_CLIN_FEATURES = [f"clin_{name}" for name in lexicon.SUBSCALE_NAMES] + \
    ["clin_total_rate", "clin_coverage"]


@_block("clinical", _CLIN_FEATURES)
def _clinical_block(doc, ctx: Context) -> dict[str, float]:
    counts = {name: 0 for name in lexicon.SUBSCALE_NAMES}
    seen: set[str] = set()
    for token in doc:
        if not token.is_alpha:
            continue
        subscale = lexicon.subscale_of(token.text, token.lemma_, token.pos_)
        if subscale is not None:
            counts[subscale] += 1
            seen.add(lexicon.fold(token.lemma_))
    features = {f"clin_{name}": ctx.rate(count) for name, count in counts.items()}
    features["clin_total_rate"] = ctx.rate(sum(counts.values()))
    features["clin_coverage"] = _ratio(len(seen), len(lexicon.CLINICAL_LEXICON))
    return features


# --- E: morphosyntax -----------------------------------------------------

@_block("morphosyntax", ["pos_pron_rate", "pos_verb_rate", "pos_adj_rate",
                         "pos_adv_rate", "pos_noun_rate", "pos_conj_rate",
                         "tense_past_rate", "tense_pres_rate", "tense_fut_rate",
                         "mood_subjunctive_rate", "modal_rate", "intensifier_rate"])
def _morphosyntax_block(doc, ctx: Context) -> dict[str, float]:
    pos_counts = {"PRON": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "NOUN": 0, "CONJ": 0}
    tense = {"Past": 0, "Pres": 0, "Fut": 0}
    subjunctive = modals = intensifiers = 0

    for token in doc:
        if not token.is_alpha:
            continue
        if token.pos_ in ("VERB", "AUX"):
            pos_counts["VERB"] += 1
        elif token.pos_ in ("CCONJ", "SCONJ"):
            pos_counts["CONJ"] += 1
        elif token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

        if token.morph.get("VerbForm") == ["Fin"]:
            for value in token.morph.get("Tense"):
                if value in tense:
                    tense[value] += 1
            if token.morph.get("Mood") == ["Sub"]:
                subjunctive += 1
        if lex.is_modal(token.lemma_):
            modals += 1
        if lex.is_intensifier(token.text):
            intensifiers += 1

    modals += lex.count_modal_phrases(ctx.text)
    return {
        "pos_pron_rate": ctx.rate(pos_counts["PRON"]),
        "pos_verb_rate": ctx.rate(pos_counts["VERB"]),
        "pos_adj_rate": ctx.rate(pos_counts["ADJ"]),
        "pos_adv_rate": ctx.rate(pos_counts["ADV"]),
        "pos_noun_rate": ctx.rate(pos_counts["NOUN"]),
        "pos_conj_rate": ctx.rate(pos_counts["CONJ"]),
        "tense_past_rate": ctx.rate(tense["Past"]),
        "tense_pres_rate": ctx.rate(tense["Pres"]),
        "tense_fut_rate": ctx.rate(tense["Fut"]),
        # The subjunctive marks the hypothetical and the wished-for. Spanish
        # has a mood for exactly the register worry lives in, which no feature
        # ported from English can offer.
        "mood_subjunctive_rate": ctx.rate(subjunctive),
        "modal_rate": ctx.rate(modals),
        "intensifier_rate": ctx.rate(intensifiers),
    }


# --- F: lexical richness -------------------------------------------------

def _mattr(lemmas: tuple[str, ...], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio.

    Plain TTR falls as a text gets longer, so over answers ranging from 12 to
    1577 words it would rank speakers by how much they said. A fixed window
    measures the same quantity at the same length for everyone.
    """
    if not lemmas:
        return 0.0
    if len(lemmas) <= window:
        return len(set(lemmas)) / len(lemmas)
    ratios = [len(set(lemmas[i:i + window])) / window
              for i in range(len(lemmas) - window + 1)]
    return float(np.mean(ratios))


def _mtld_pass(lemmas: tuple[str, ...], threshold: float) -> float:
    factors, types, count = 0.0, set(), 0
    for lemma in lemmas:
        types.add(lemma)
        count += 1
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count > 0:
        remaining = len(types) / count
        if remaining < 1.0:
            factors += (1 - remaining) / (1 - threshold)
    return len(lemmas) / factors if factors > 0 else float(len(lemmas))


def _mtld(lemmas: tuple[str, ...], threshold: float = MTLD_THRESHOLD) -> float:
    """Measure of textual lexical diversity, averaged over both directions.

    Reported alongside MATTR because the two fail differently: MATTR is blind
    to anything beyond its window, MTLD is unstable on very short texts. The
    diagnostic reports how far they agree here.
    """
    if not lemmas:
        return 0.0
    return float(np.mean([_mtld_pass(lemmas, threshold),
                          _mtld_pass(tuple(reversed(lemmas)), threshold)]))


@_block("richness", ["mattr_50", "mtld", "hapax_rate", "mean_word_len",
                     "lexical_density", "fernandez_huerta"])
def _richness_block(doc, ctx: Context) -> dict[str, float]:
    alpha = [t for t in doc if t.is_alpha]
    counts: dict[str, int] = {}
    for lemma in ctx.lemmas:
        counts[lemma] = counts.get(lemma, 0) + 1
    hapax = sum(1 for n in counts.values() if n == 1)
    content = sum(1 for t in alpha if t.pos_ in _CONTENT_POS)
    syllables = sum(lex.count_syllables(t.text) for t in alpha)

    words_per_sentence = _ratio(ctx.n_words, ctx.n_sentences)
    syllables_per_word = _ratio(syllables, ctx.n_words)
    return {
        "mattr_50": _mattr(ctx.lemmas),
        "mtld": _mtld(ctx.lemmas),
        # Diversity is measured over lemmas, not surface forms: Spanish
        # inflection would otherwise count "siento", "sentía" and "sentido" as
        # three types and read conjugation as vocabulary. Same reason
        # encoders.py lemmatizes before TF-IDF.
        "hapax_rate": _ratio(hapax, ctx.n_words),
        "mean_word_len": _ratio(sum(len(t.text) for t in alpha), ctx.n_words),
        "lexical_density": _ratio(content, ctx.n_words),
        # Fernández-Huerta, the Spanish adaptation of Flesch. Higher is easier.
        "fernandez_huerta": 206.84 - 60.0 * syllables_per_word - 1.02 * words_per_sentence,
    }


# --- G: spoken discourse -------------------------------------------------

@_block("discourse", ["discourse_marker_rate", "hesitation_rate", "repetition_rate",
                      "mean_sentence_len", "sentence_len_sd"])
def _discourse_block(doc, ctx: Context) -> dict[str, float]:
    markers = sum(1 for t in doc if t.is_alpha and lex.is_discourse_marker(t.text))
    markers += lex.count_discourse_phrases(ctx.text)

    repetitions = sum(1 for a, b in zip(ctx.lemmas, ctx.lemmas[1:]) if a == b)
    lengths = ctx.sentence_lengths or (0,)
    return {
        "discourse_marker_rate": ctx.rate(markers),
        # Whisper writes an ellipsis where the speaker trails off or restarts.
        # It is the only hesitation evidence left in the transcript: filler
        # sounds do not survive the model's normalization.
        "hesitation_rate": ctx.rate(len(lex.ELLIPSIS_RE.findall(ctx.text))),
        "repetition_rate": ctx.rate(repetitions),
        "mean_sentence_len": float(np.mean(lengths)),
        "sentence_len_sd": float(np.std(lengths)),
    }


# --- Affect: Spanish affective norms -------------------------------------

_AFFECT_FEATURES = (
    ["val_mean", "val_sd", "val_min", "val_p05", "val_low_rate", "val_high_rate",
     "aro_mean", "aro_max", "aro_high_rate"]
    + [f"emo_{emotion}_{statistic}"
       for emotion in affective_norms.EMOTIONS for statistic in ("mean", "max")]
    + ["affect_coverage", "emo_coverage"]
)


@_block("affect", _AFFECT_FEATURES)
def _affect_block(doc, ctx: Context) -> dict[str, float]:
    """Valence, arousal and five discrete emotions, from ratings Spanish
    speakers gave to Spanish words (src/affective_norms.py).

    Three statistics per construct, because they answer different questions
    and the training set says all three earn their place. A mean gives the
    overall tone but flattens the extremes: one very dark word in four
    hundred neutral ones leaves it unmoved. An extreme keeps that word --
    `emo_fear_max` reaches rho = +0.43 against GAD-7 on the training
    participants, the strongest single feature measured anywhere in this
    study, and it survives partialling out how much the person said. A rate
    says how much of the discourse sits past a cut, which for valence adds
    something the minimum does not (the two correlate at only -0.19, and each
    is significant given the other).

    Extremes are known to grow with the number of words they are taken over;
    the partial correlations say the signal is not only that, but the arm
    carries no length feature to adjust with, so E12 is where they can be
    read cleanly against one.
    """
    norms = affective_norms.load_norms()
    valences, arousals = [], []
    emotions = {emotion: [] for emotion in affective_norms.EMOTIONS}
    n_content = n_emotion = 0

    for token in doc:
        if not token.is_alpha or token.pos_ not in _AFFECT_POS:
            continue
        n_content += 1
        forms = (token.lemma_, token.text)
        if not norms.pos_agrees(token.pos_, *forms):
            continue
        valence = norms.valence_of(*forms)
        if valence is not None:
            valences.append(valence)
            arousals.append(norms.arousal_of(*forms))
        rated = norms.emotions_of(*forms)
        if rated is not None:
            n_emotion += 1
            for emotion, score in rated.items():
                emotions[emotion].append(score)

    value = np.array(valences)
    agitation = np.array(arousals)
    features = {
        "val_mean": _stat(value, np.mean, affective_norms.VALENCE_NEUTRAL),
        "val_sd": _stat(value, np.std, 0.0),
        "val_min": _stat(value, np.min, affective_norms.VALENCE_NEUTRAL),
        "val_p05": _stat(value, lambda a: np.percentile(a, 5),
                         affective_norms.VALENCE_NEUTRAL),
        "val_low_rate": _stat(value, lambda a: (a < affective_norms.VALENCE_LOW).mean(), 0.0),
        "val_high_rate": _stat(value, lambda a: (a > affective_norms.VALENCE_HIGH).mean(), 0.0),
        "aro_mean": _stat(agitation, np.mean, affective_norms.AROUSAL_NEUTRAL),
        "aro_max": _stat(agitation, np.max, affective_norms.AROUSAL_NEUTRAL),
        "aro_high_rate": _stat(agitation,
                               lambda a: (a > affective_norms.AROUSAL_HIGH).mean(), 0.0),
    }
    # Emitted in the order _AFFECT_FEATURES declares: the emotions before the
    # two coverage columns. A block whose keys arrive in a different order
    # than it declared is a contract break the tests fail on, even though the
    # frame builders reindex by name and would not notice.
    for emotion in affective_norms.EMOTIONS:
        scores = np.array(emotions[emotion])
        features[f"emo_{emotion}_mean"] = _stat(scores, np.mean,
                                                affective_norms.EMOTION_NEUTRAL)
        features[f"emo_{emotion}_max"] = _stat(scores, np.max,
                                               affective_norms.EMOTION_NEUTRAL)
    features["affect_coverage"] = _ratio(len(valences), n_content)
    features["emo_coverage"] = _ratio(n_emotion, n_content)
    return features


def _stat(values: np.ndarray, statistic: Callable, empty: float) -> float:
    """A statistic over the rated words, or the declared neutral when none of
    a text's words carry a rating. Never NaN, and never 0.0 by accident: on a
    1-9 valence scale whose floor is 1.15, a zero would read as the most
    distressed answer in the corpus rather than as a missing measurement.
    The coverage features are what say it was not measured.
    """
    return float(statistic(values)) if len(values) else float(empty)


# --- Feature sets --------------------------------------------------------

@dataclass(frozen=True)
class FeatureSet:
    """One arm's worth of features: which blocks, which of them get a spread
    across a participant's answers, and whether the four length measures are
    included.

    The affective set leaves the length measures out on purpose. It is
    defined by being about affect, and `n_words_total` is not; the extremes
    it carries are the ones that would most like a length control, which is
    exactly the comparison E12 exists to make.
    """

    version: str
    blocks: tuple[str, ...]
    sd_features: tuple[str, ...]
    structural: bool


LEXICAL_BLOCKS = ("person", "negation", "clinical", "morphosyntax",
                  "richness", "discourse")

FEATURE_SETS: dict[str, FeatureSet] = {
    # E10. The version string is what the already-logged runs recorded, so it
    # must not change: results/frozen/*/*/experiment_log.csv carries it as the
    # encoder_model of every E10 row, and the cached matrices as their signature.
    "lexical": FeatureSet(FEATURE_SET_VERSION, LEXICAL_BLOCKS, SD_FEATURES, True),
    # E11: affect alone.
    "affect": FeatureSet("affect-v1", ("affect",), AFFECT_SD_FEATURES, False),
    # E12: both, which is the only arm that can say whether the two sources
    # are complementary or say the same thing twice.
    "lexical-affect": FeatureSet("lexical-affect-v1", LEXICAL_BLOCKS + ("affect",),
                                 SD_FEATURES + AFFECT_SD_FEATURES, True),
}

DEFAULT_FEATURE_SET = "lexical"

STRUCTURAL_FEATURES = ("n_words_total", "n_answers",
                       "words_per_answer_mean", "words_per_answer_sd")

_BLOCKS_BY_NAME = {block.name: block for block in BLOCKS}


def feature_set(name: str = DEFAULT_FEATURE_SET) -> FeatureSet:
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; expected one of "
                         f"{sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name]


def _blocks_of(spec: FeatureSet) -> list[Block]:
    return [_BLOCKS_BY_NAME[name] for name in spec.blocks]


# --- Public API ----------------------------------------------------------

def answer_features(name: str = DEFAULT_FEATURE_SET) -> tuple[str, ...]:
    """Per-answer column order for one feature set, in block order."""
    return tuple(feature for block in _blocks_of(feature_set(name))
                 for feature in block.features)


def participant_features(name: str = DEFAULT_FEATURE_SET) -> tuple[str, ...]:
    """Per-participant column order: the answer features averaged, the
    declared spreads across a person's answers, then the length measures for
    the sets that carry them."""
    spec = feature_set(name)
    return (answer_features(name)
            + tuple(f"{feature}_sd" for feature in spec.sd_features)
            + (STRUCTURAL_FEATURES if spec.structural else ()))


# The default set's names, kept as module constants because callers written
# before there was more than one feature set read them directly.
ANSWER_FEATURES: tuple[str, ...] = answer_features()
PARTICIPANT_FEATURES: tuple[str, ...] = participant_features()


def feature_names(level: str = config.GRANULARITY_PARTICIPANT,
                  name: str = DEFAULT_FEATURE_SET) -> tuple[str, ...]:
    """Column order for one granularity and feature set. Derived from the
    block registry, so it cannot drift from what the blocks actually return.
    """
    if level == config.GRANULARITY_PARTICIPANT:
        return participant_features(name)
    if level == config.GRANULARITY_QUESTION:
        return answer_features(name)
    raise ValueError(f"Unknown granularity {level!r}")


def extract_answer(text: str, nlp=None,
                   name: str = DEFAULT_FEATURE_SET) -> dict[str, float]:
    """Features of one answer. Keys are exactly answer_features(name)."""
    nlp = nlp or get_nlp()
    return _extract_doc(nlp(lex.strip_placeholders(text or "")), name)


def _extract_doc(doc, name: str = DEFAULT_FEATURE_SET) -> dict[str, float]:
    ctx = _context(doc)
    features: dict[str, float] = {}
    for block in _blocks_of(feature_set(name)):
        features.update(block.fn(doc, ctx))
    return features


def extract_participant(texts, nlp=None,
                        name: str = DEFAULT_FEATURE_SET) -> dict[str, float]:
    """One participant's features: their answers averaged, the spread of the
    declared features across those answers, and, for the sets that carry
    them, four length measures.

    Standard deviations use ddof=0, so a participant with a single usable
    answer gets 0.0 rather than NaN. That is a statement about this person's
    spread being unmeasurable, not about it being zero, but the matrix has to
    hold a number and the count of answers is itself a feature.
    """
    spec = feature_set(name)
    columns = participant_features(name)
    answers = [t for t in texts if isinstance(t, str) and t.strip()]
    if not answers:
        return {column: 0.0 for column in columns}

    nlp = nlp or get_nlp()
    docs = list(nlp.pipe(lex.strip_placeholders(t) for t in answers))
    per_answer = [_extract_doc(doc, name) for doc in docs]
    word_counts = [_context(doc).n_words for doc in docs]

    features = {feature: float(np.mean([a[feature] for a in per_answer]))
                for feature in answer_features(name)}
    features.update({f"{feature}_sd": float(np.std([a[feature] for a in per_answer]))
                     for feature in spec.sd_features})
    if spec.structural:
        features.update({
            "n_words_total": float(sum(word_counts)),
            "n_answers": float(len(answers)),
            "words_per_answer_mean": float(np.mean(word_counts)),
            "words_per_answer_sd": float(np.std(word_counts)),
        })
    return features


def extract_frame(frame: pd.DataFrame,
                  questions: list[str] = config.QUESTIONS,
                  name: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    """Participant-level feature matrix aligned to frame.index."""
    nlp = get_nlp()
    rows = [extract_participant([row[q] for q in questions], nlp, name)
            for _, row in frame.iterrows()]
    return pd.DataFrame(rows, index=frame.index,
                        columns=list(participant_features(name)))


def extract_question_frame(frame: pd.DataFrame, text_column: str = "text",
                           name: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    """Question-level feature matrix, one row per answer, aligned to
    frame.index. Takes the frame pipeline.expand_to_questions produces.
    """
    nlp = get_nlp()
    docs = nlp.pipe(lex.strip_placeholders(t) for t in frame[text_column])
    rows = [_extract_doc(doc, name) for doc in docs]
    return pd.DataFrame(rows, index=frame.index, columns=list(answer_features(name)))
