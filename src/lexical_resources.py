"""Closed word lists for the lexical arm (E10), and the Spanish-specific
helpers its features need.

Only the lists that spaCy's morphology cannot supply live here. Person,
number, tense, mood and negation are read from `token.morph` instead of
matched against a list, because a list of pronouns undercounts Spanish
systematically: the language is pro-drop, so "estoy cansada" carries a first
person that no pronoun in the sentence marks. What remains for a list is the
vocabulary that is defined by its discourse function rather than by its
morphology: absolutist terms, discourse markers, intensifiers and modals.

Everything is matched accent- and case-folded through `lexicon.fold`, so
"jamas" and "jamás" are the same entry, and the transcription's own accent
inconsistencies do not decide whether a feature fires.

Nothing here is adapted from social-media feature sets. Mentions, hashtags,
emoji and retweet markers do not occur in a transcribed spoken answer, and a
feature that is constant at zero costs a column and contributes nothing;
tests/test_lexical_features.py fails on any feature with no variance over the
training participants.
"""
from __future__ import annotations

import re

from src.lexicon import fold

# Words that admit no degree. Al-Mosaiwi & Johnstone (2018) found this class
# separates depression and anxiety forums better than negative-emotion words,
# which is why it is its own feature rather than part of the affective count.
ABSOLUTIST = frozenset({
    "siempre", "nunca", "jamás", "todo", "todos", "toda", "todas", "nada",
    "nadie", "ninguno", "ninguna", "ningún", "completamente", "totalmente",
    "absolutamente", "constantemente", "definitivamente", "entero", "entera",
})

# Explicit negation beyond what Polarity=Neg marks. spaCy tags "no" and a few
# others; the negative quantifiers and adverbs below carry negation lexically
# and are not consistently marked in the morphology.
NEGATION_LEMMAS = frozenset({
    "no", "nunca", "jamás", "nada", "nadie", "ninguno", "ningún", "tampoco", "ni",
})

# Single-token discourse markers of spoken Spanish. These survive Whisper's
# transcription, unlike filler sounds: the corpus holds 3 occurrences of "eh"
# in 544 answers, so a filler feature would be constant at zero.
DISCOURSE_MARKERS = frozenset({
    "bueno", "pues", "entonces", "digamos", "claro", "obviamente", "realmente",
})

# Multi-word markers, matched on the folded text with word boundaries, since
# the tokenizer splits them and no single token identifies them.
DISCOURSE_PHRASES = (
    "o sea", "osea", "es que", "como que", "la verdad", "de pronto",
    "por ejemplo", "en realidad", "digamos que", "nada mas", "asi que",
)

INTENSIFIERS = frozenset({
    "muy", "bastante", "demasiado", "súper", "super", "tan", "tanto",
    "totalmente", "completamente", "bien", "harto", "full",
})

# Verbs of ability, obligation and volition. Read as perceived constraint:
# how much of what is said is framed as something one can, must or wants to
# do, rather than as something one does.
MODAL_LEMMAS = frozenset({
    "poder", "deber", "querer", "necesitar", "intentar", "tratar",
})

# The periphrastic obligations are two tokens, so no lemma can match them and
# they would be a dead entry in the set above. They are counted on the text.
MODAL_PHRASES = ("tener que", "hay que", "toca que")

# Anonymization placeholders introduced during transcription review. They are
# not language: left in, spaCy tokenizes the brackets as punctuation and the
# body as a noun, which inflates the noun rate and the word count.
PLACEHOLDER_RE = re.compile(r"\[[A-ZÁÉÍÓÚÑ_]+\]")

# Whisper writes an ellipsis where the speaker trails off or self-interrupts.
# It is the only hesitation evidence that survives into the transcript.
ELLIPSIS_RE = re.compile(r"\.\.\.|…")

_FOLDED_ABSOLUTIST = frozenset(fold(w) for w in ABSOLUTIST)
_FOLDED_NEGATION = frozenset(fold(w) for w in NEGATION_LEMMAS)
_FOLDED_MARKERS = frozenset(fold(w) for w in DISCOURSE_MARKERS)
_FOLDED_INTENSIFIERS = frozenset(fold(w) for w in INTENSIFIERS)
_FOLDED_MODALS = frozenset(fold(w) for w in MODAL_LEMMAS)

def _phrase_patterns(phrases) -> tuple:
    return tuple(re.compile(r"\b" + re.escape(fold(p)) + r"\b") for p in phrases)


_PHRASE_RES = _phrase_patterns(DISCOURSE_PHRASES)
_MODAL_PHRASE_RES = _phrase_patterns(MODAL_PHRASES)


def is_absolutist(form: str, lemma: str | None = None) -> bool:
    return fold(form) in _FOLDED_ABSOLUTIST or (
        lemma is not None and fold(lemma) in _FOLDED_ABSOLUTIST)


def is_negation(form: str, lemma: str | None = None) -> bool:
    return fold(form) in _FOLDED_NEGATION or (
        lemma is not None and fold(lemma) in _FOLDED_NEGATION)


def is_discourse_marker(form: str) -> bool:
    return fold(form) in _FOLDED_MARKERS


def is_intensifier(form: str) -> bool:
    return fold(form) in _FOLDED_INTENSIFIERS


def is_modal(lemma: str) -> bool:
    return fold(lemma) in _FOLDED_MODALS


def count_discourse_phrases(text: str) -> int:
    """Occurrences of the multi-word markers in one answer."""
    folded = fold(text)
    return sum(len(pattern.findall(folded)) for pattern in _PHRASE_RES)


def count_modal_phrases(text: str) -> int:
    """Occurrences of the periphrastic obligations ("tener que", "hay que")."""
    folded = fold(text)
    return sum(len(pattern.findall(folded)) for pattern in _MODAL_PHRASE_RES)


def strip_placeholders(text: str) -> str:
    return PLACEHOLDER_RE.sub(" ", text)


# --- Spanish syllabification ------------------------------------------

_VOWELS = set("aeiouáéíóúüàèìòù")
_STRONG = set("aeoáéó")
_BREAKING = set("íú")   # an accented weak vowel breaks what would be a diphthong


def _syllables_in_group(group: str) -> int:
    """Vowels in contact form one syllable unless they are two strong vowels
    (hiatus: "ca-er") or the weak one carries the stress ("dí-a")."""
    syllables = 1
    for first, second in zip(group, group[1:]):
        if (first in _STRONG and second in _STRONG) or first in _BREAKING or second in _BREAKING:
            syllables += 1
    return syllables


def count_syllables(word: str) -> int:
    """Syllables of one Spanish word, by vowel groups.

    Spanish syllabification is regular enough that a rule over vowel groups is
    accurate without a dictionary, which is what the Fernández-Huerta
    readability score needs. Never returns 0, so a token of consonants alone
    ("mmm") still counts as a syllable rather than making the words-per-
    syllable ratio undefined.
    """
    lowered = word.lower()
    syllables, index, length = 0, 0, len(lowered)
    while index < length:
        if lowered[index] in _VOWELS:
            end = index
            while end < length and lowered[end] in _VOWELS:
                end += 1
            syllables += _syllables_in_group(lowered[index:end])
            index = end
        else:
            index += 1
    return max(syllables, 1)
