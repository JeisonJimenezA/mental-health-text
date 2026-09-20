"""Clinical lexicon for the depression and anxiety vocabulary of this domain.

Lemmas from the PHQ-9 and GAD-7 items (interest, mood, sleep, energy,
appetite, guilt, concentration, psychomotor change, self-harm; nerves,
worry, relaxation, restlessness, irritability, fear), plus everyday words
people use for the same states. Assembled for this study from the two
questionnaires, not a validated resource; it is used to describe how
tokenizers treat the domain (scripts/vocabulary_diagnostic.py), optionally
to mask these words more often during MLM adaptation (src/mlm_adaptation.py),
and as the symptom-rate features of the lexical arm
(src/lexical_features.py).

The terms are held as CLINICAL_SUBSCALES, one group per symptom, and
CLINICAL_LEXICON is their union. The partition exists because a single
count of "clinical words" says only that someone talked about their state,
while a rate per symptom is what can be read against a PHQ-9 total item by
item. Every term belongs to exactly one subscale, and the union is
byte-identical to the flat set that preceded it: the vocabulary diagnostic
already published under results/diagnostics/vocabulary/ and the masking
scheme of the adapted checkpoints under models/ were both computed from
that set, and changing its membership would silently invalidate them.
tests/test_lexical_features.py locks both properties.

Matching is accent- and case-insensitive, on either the lemma or the surface
form, so inflected forms ("estresada", "dormí") count.
"""
import unicodedata

# One group per PHQ-9 / GAD-7 symptom. Everyday words sit with the item they
# describe ("bajón" with mood, "agobio" with stress), because what matters
# downstream is the state being reported, not which questionnaire the word
# came from. Two groups have no item of their own: `estres`, which the
# questionnaires do not ask about but which dominates how students describe
# this period, and `soledad`, which is social rather than symptomatic.
CLINICAL_SUBSCALES: dict[str, frozenset] = {
    # --- PHQ-9 ---
    "anhedonia": frozenset({"interés", "placer", "desmotivado", "motivación"}),
    "animo": frozenset({"decaído", "deprimido", "depresión", "esperanza", "desesperanza",
                        "triste", "tristeza", "llorar", "bajón", "desánimo", "vacío"}),
    "sueno": frozenset({"dormir", "sueño", "insomnio"}),
    "energia": frozenset({"cansado", "cansancio", "energía", "agotado", "agotamiento"}),
    "apetito": frozenset({"apetito", "comer"}),
    "culpa": frozenset({"fracaso", "culpa", "culpable", "autoestima"}),
    "concentracion": frozenset({"concentrar", "concentración"}),
    "psicomotor": frozenset({"lento", "inquieto", "inquietud"}),
    "autolesion": frozenset({"muerte", "morir", "lastimar", "suicidio", "autolesión"}),
    # --- GAD-7 ---
    # "relajar" is GAD-7's own item ("no poder relajarse"); it sits with the
    # nerves group rather than alone, since one term cannot carry a rate.
    "nervios": frozenset({"nervioso", "nervio", "ansioso", "ansiedad",
                          "angustia", "angustiado", "relajar"}),
    "preocupacion": frozenset({"preocupar", "preocupación", "preocupado", "sobrepensar"}),
    "irritabilidad": frozenset({"irritable", "irritabilidad", "irritar"}),
    "miedo": frozenset({"miedo", "temor", "pánico", "terrible"}),
    # --- no questionnaire item of their own ---
    "estres": frozenset({"estrés", "estresado", "estresar", "agobiado", "agobio",
                         "abrumado", "frustrado", "frustración"}),
    "soledad": frozenset({"solo", "soledad"}),
}

SUBSCALE_NAMES = tuple(CLINICAL_SUBSCALES)

CLINICAL_LEXICON = frozenset().union(*CLINICAL_SUBSCALES.values())

# Surface forms the lemmatizer does not resolve to a lexicon entry.
#
# es_core_news_sm lemmatizes by lookup, and it has two systematic gaps on
# exactly this vocabulary. Stem-changing verbs keep their diphthong, so
# "duermo" lemmatizes to "duermo" rather than "dormir" -- and "duermo" occurs
# 90 times in the training interviews (results/diagnostics/vocabulary/).
# Feminine adjectives are resolved inconsistently: "agotada" reaches
# "agotado", while "estresada" is tagged NOUN and left as itself. Since every
# adjective in the lexicon above is listed in the masculine, the second gap
# under-counts women's answers specifically, which is a bias and not just
# noise.
#
# These forms are consulted by subscale_of only. CLINICAL_LEXICON and
# is_clinical stay exactly as they were: the vocabulary diagnostic and the
# adapted checkpoints under models/ were computed from that set, and widening
# it would silently invalidate both.
CLINICAL_INFLECTIONS: dict[str, str] = {}


def _add_inflections(subscale: str, *forms: str) -> None:
    for form in forms:
        CLINICAL_INFLECTIONS[form] = subscale


_add_inflections("anhedonia", "intereses", "desmotivada", "desmotivados", "desmotivadas")
_add_inflections("animo", "decaída", "decaídos", "decaídas", "deprimida", "deprimidos",
                 "deprimidas", "depresiones", "tristes", "lloro", "lloras", "llora",
                 "lloro", "lloraba", "bajones", "vacía", "vacíos", "vacías")
_add_inflections("sueno", "duermo", "duermes", "duerme", "duermen", "durmiendo",
                 "dormía", "dormido", "sueños")
_add_inflections("energia", "cansada", "cansados", "cansadas", "agotada", "agotados",
                 "agotadas", "energías")
_add_inflections("apetito", "como", "comes", "come", "comen", "comiendo", "comía")
_add_inflections("culpa", "culpables", "fracasos", "culpas")
_add_inflections("concentracion", "concentro", "concentras", "concentra", "concentrarme",
                 "concentraciones")
_add_inflections("psicomotor", "lenta", "lentos", "lentas", "inquieta", "inquietos",
                 "inquietas")
_add_inflections("autolesion", "muero", "mueres", "muere", "mueren", "muriendo", "muertes")
_add_inflections("nervios", "nerviosa", "nerviosos", "nerviosas", "ansiosa", "ansiosos",
                 "ansiosas", "angustiada", "angustiados", "angustiadas", "nervios",
                 "relajo", "relaja", "relajarme")
_add_inflections("preocupacion", "preocupada", "preocupados", "preocupadas", "preocupo",
                 "preocupa", "preocupan", "preocuparme", "sobrepienso")
_add_inflections("irritabilidad", "irritables", "irrito", "irrita")
_add_inflections("miedo", "miedos", "terribles")
_add_inflections("estres", "estresada", "estresados", "estresadas", "agobiada",
                 "agobiados", "agobiadas", "abrumada", "abrumados", "abrumadas",
                 "frustrada", "frustrados", "frustradas", "estreso", "estresa",
                 "estresarme")
_add_inflections("soledad", "sola", "solos", "solas")

# Forms that are homographs of a much more frequent non-clinical word. The
# written accent that used to separate "sólo" (only) from "solo" (alone) was
# abolished, and "como" is both "I eat" and "as/like". Counting them blind
# would make those two subscales mostly noise, so they only count under the
# part of speech that carries the clinical reading.
_POS_GUARD: dict[str, frozenset] = {
    "solo": frozenset({"ADJ"}),
    "sola": frozenset({"ADJ"}),
    "solos": frozenset({"ADJ"}),
    "solas": frozenset({"ADJ"}),
    "como": frozenset({"VERB", "AUX"}),
    "come": frozenset({"VERB", "AUX"}),
    "comes": frozenset({"VERB", "AUX"}),
    "comen": frozenset({"VERB", "AUX"}),
    "lento": frozenset({"ADJ"}),
    "lenta": frozenset({"ADJ"}),
    "lentos": frozenset({"ADJ"}),
    "lentas": frozenset({"ADJ"}),
}


def fold(text: str) -> str:
    """Lowercase without accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))


_FOLDED = frozenset(fold(term) for term in CLINICAL_LEXICON)

# Folded lookup from term to its subscale, so a matched token is attributed in
# one dict access instead of a scan over every group. The inflections are
# merged in here, and only here: CLINICAL_LEXICON above stays the frozen set.
_FOLDED_SUBSCALE = {fold(term): name
                    for name, terms in CLINICAL_SUBSCALES.items()
                    for term in terms}
_FOLDED_SUBSCALE.update({fold(form): name for form, name in CLINICAL_INFLECTIONS.items()})

_FOLDED_POS_GUARD = {fold(form): tags for form, tags in _POS_GUARD.items()}


def is_clinical(form: str, lemma: str | None = None) -> bool:
    """Membership in the frozen lexicon. Unchanged, and deliberately blind to
    CLINICAL_INFLECTIONS: the published vocabulary diagnostic and the MLM
    masking of the adapted checkpoints were both computed through this
    function, so widening it would make those artifacts unreproducible.
    """
    return fold(form) in _FOLDED or (lemma is not None and fold(lemma) in _FOLDED)


def subscale_of(form: str, lemma: str | None = None, pos: str | None = None) -> str | None:
    """Which symptom group a token belongs to, or None.

    Wider than is_clinical by the inflections above, because this one feeds a
    feature that has to fire on how people actually speak rather than on a
    citation form. The lemma is tried first, since spaCy resolves "agotada" to
    "agotado" and "dormí" to "dormir"; the surface form catches what the
    lemmatizer leaves inflected, which on this model is most of the present
    tense of the stem-changing verbs.

    `pos` is required to count the homographs in _POS_GUARD and ignored for
    everything else. Passing it is how "me siento solo" is told apart from
    "solo quería decir".
    """
    candidates = [lemma, form]
    # spaCy returns a multi-word lemma when a clitic is attached, so
    # "preocuparme" comes back as "preocupar yo" and "concentrarme" as
    # "concentrar yo". The verb is the first word; without this the whole
    # string is looked up and never matches. Generalizes to the reflexive
    # forms of any lexicon verb, which the inflection list cannot enumerate.
    if lemma is not None and " " in lemma:
        candidates.insert(1, lemma.split()[0])

    for candidate in candidates:
        if candidate is None:
            continue
        folded = fold(candidate)
        hit = _FOLDED_SUBSCALE.get(folded)
        if hit is None:
            continue
        allowed = _FOLDED_POS_GUARD.get(folded)
        if allowed is not None and (pos is None or pos not in allowed):
            continue
        return hit
    return None
