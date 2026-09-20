"""Shared orchestration for the Phase B baseline comparison: building the
training frame (plain or augmented), computing/caching embeddings, and
running the search+logging loop for one encoder across seeds and targets.

Every function takes the run's TrainingConfig explicitly, and the config is
logged with each result row, so a run is reproducible from its log alone.

Used by scripts/run_phase_b.py (headless batch runs) and
notebooks/01_baseline_encoders.ipynb (interactive, one encoder per cell) so
the two never drift apart.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src import config, data, encoders, lexical_features, metrics, splits, logging_utils
from src.regression_models import search_best_model, best_of

TARGETS = [config.PRIMARY_TARGET, config.SECONDARY_TARGET]

# Named features rather than learned dimensions (src/lexical_features.py),
# as three arms that decompose one question. E10 is the lexical set alone,
# E11 the Spanish affective norms alone, E12 both. Only the third can say
# whether the two sources are complementary or say the same thing twice, and
# only by comparing it against the other two on the same held-out people.
#
# All three belong to the frozen family, like E0-E2: the representation is
# computed once and a classical regressor is fitted on top, so their rows
# land beside those and every comparison stays paired.
LEXICAL_ARM = "E10_LEXICAL"
LEXICAL_ARM_QUESTION = "E10_LEXICAL_QUESTION"
AFFECT_ARM = "E11_AFFECT"
AFFECT_ARM_QUESTION = "E11_AFFECT_QUESTION"
LEXICAL_AFFECT_ARM = "E12_LEXICAL_AFFECT"
LEXICAL_AFFECT_ARM_QUESTION = "E12_LEXICAL_AFFECT_QUESTION"

# E13-E15: each named-feature set beside a TF-IDF of the same text. The two
# kinds of evidence differ -- the named columns are counts of declared
# categories, TF-IDF is the vocabulary itself with no category imposed on it
# -- and with E10-E12 these complete every combination of the three sources:
#
#   lexical            E10        lexical + affect            E12
#   affect             E11        lexical + tfidf             E13
#   tfidf              E0         affect  + tfidf             E14
#                                 lexical + affect + tfidf    E15
#
# E0 is the one cell measured elsewhere (notebook 01) and under the full
# nine-model set, so it is not run here and does not compare row for row.
LEXICAL_TFIDF_ARM = "E13_LEXICAL_TFIDF"
LEXICAL_TFIDF_ARM_QUESTION = "E13_LEXICAL_TFIDF_QUESTION"
AFFECT_TFIDF_ARM = "E14_AFFECT_TFIDF"
AFFECT_TFIDF_ARM_QUESTION = "E14_AFFECT_TFIDF_QUESTION"
LEXICAL_AFFECT_TFIDF_ARM = "E15_LEXICAL_AFFECT_TFIDF"
LEXICAL_AFFECT_TFIDF_ARM_QUESTION = "E15_LEXICAL_AFFECT_TFIDF_QUESTION"

# Wider than E0's window. Trigrams are what a category counter cannot see: a
# negated state ("no me siento bien") is three tokens and no single feature
# of the lexical block encodes it.
LEXICAL_TFIDF_NGRAM = (1, 3)

# Which feature set each arm draws from, and therefore which cache file and
# which signature it carries.
FEATURE_SET_OF_ARM = {
    LEXICAL_ARM: "lexical", LEXICAL_ARM_QUESTION: "lexical",
    AFFECT_ARM: "affect", AFFECT_ARM_QUESTION: "affect",
    LEXICAL_AFFECT_ARM: "lexical-affect", LEXICAL_AFFECT_ARM_QUESTION: "lexical-affect",
}

# Which named-feature arm each TF-IDF combination sits on top of. The base
# arm's cached matrix is reused rather than recomputed, so E13 and E10 are
# guaranteed to carry the identical lexical columns and the contrast between
# them is the vectorizer and nothing else.
TFIDF_BASE_ARM = {
    LEXICAL_TFIDF_ARM: LEXICAL_ARM,
    AFFECT_TFIDF_ARM: AFFECT_ARM,
    LEXICAL_AFFECT_TFIDF_ARM: LEXICAL_AFFECT_ARM,
    LEXICAL_TFIDF_ARM_QUESTION: LEXICAL_ARM_QUESTION,
    AFFECT_TFIDF_ARM_QUESTION: AFFECT_ARM_QUESTION,
    LEXICAL_AFFECT_TFIDF_ARM_QUESTION: LEXICAL_AFFECT_ARM_QUESTION,
}


def lexical_model_tag(feature_set: str = "lexical") -> str:
    """What an arm records as the model that produced its features. The
    feature-set version is part of it, so a run made before a feature was
    added cannot be read as if it had been made after.
    """
    return f"lexical:{lexical_features.feature_set(feature_set).version}"


# Kept for callers written when E10 was the only such arm.
LEXICAL_MODEL = lexical_model_tag("lexical")

# Every arm whose features are named rather than learned.
NAMED_FEATURE_ARMS = frozenset(FEATURE_SET_OF_ARM) | frozenset(TFIDF_BASE_ARM)


def family_of(arm: str) -> str:
    """Which results family an arm's rows belong to.

    Derived from the arm rather than passed in at the call site, so a row
    cannot be written under one family by the notebook and another by the
    batch script. The named-feature arms are procedurally frozen arms, and
    are logged apart so that study is one tree of its own.
    """
    return config.FAMILY_LEXICAL if arm in NAMED_FEATURE_ARMS else config.FAMILY_FROZEN


def lexical_results_dir(cfg: config.TrainingConfig,
                        question_level: bool = False) -> Path:
    """Where the named-feature arms write, for both their logs and caches.

    The cache belongs beside the log it produced: one directory per family
    keeps a matrix from being read by a run that is recorded somewhere else,
    which is how a stale cache goes unnoticed.
    """
    return config.results_dir(
        cfg.augmentation, config.FAMILY_LEXICAL,
        config.GRANULARITY_QUESTION if question_level else config.GRANULARITY_PARTICIPANT)


def checkpoint_label(model_name) -> str:
    """How a checkpoint is recorded on a result row.

    A hub id passes through unchanged. A local path is made relative to the
    experiment root, with forward slashes: `models/tsdae_e5_small/ep05` says
    which checkpoint produced a row on any machine, where an absolute path
    says it only on the one that wrote it and pins that machine's layout
    into a file meant to be shared.
    """
    text = str(model_name)
    try:
        return Path(text).resolve().relative_to(config.EXPERIMENT_ROOT).as_posix()
    except (ValueError, OSError):
        return text


def model_tag(model_name: str) -> str:
    """Short identifier for a checkpoint, used to name both the arm and its
    embedding cache so the two can never disagree about which model produced
    the vectors. Accepts hub ids and local paths alike.
    """
    name = str(model_name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower().removeprefix("multilingual_")


def e5_arm(model_name: str, question_level: bool = False) -> str:
    """Arm id for one sentence-transformer checkpoint, e.g. E2_E5SMALL.

    Derived from the checkpoint rather than hard-coded, so each model gets
    its own log rows and its own embedding cache instead of one silently
    overwriting another's results.
    """
    return f"E2_{model_tag(model_name).replace('_', '').upper()}" + \
        ("_QUESTION" if question_level else "")


def _e5_cache_file(model_name: str, question_level: bool) -> str:
    return f"{model_tag(model_name)}{'_questions' if question_level else ''}.npy"


def e5_models_of(cfg: config.TrainingConfig, e5_models=None) -> list[str]:
    return list(e5_models) if e5_models else [cfg.sentence_transformer_model]


def mean_pooled_models(cfg: config.TrainingConfig, question_level: bool = False) -> dict:
    """{arm id: checkpoint} for masked-language models embedded the way E1
    embeds BETO: mean pooling over the last hidden state, no prefix, sliding
    windows. Empty when the arm is switched off in the configuration.
    """
    if not cfg.mental_es_model:
        return {}
    suffix = "_QUESTION" if question_level else ""
    return {f"E7_ROBERTA_MENTAL{suffix}": cfg.mental_es_model}


def search_label(cfg: config.TrainingConfig) -> str:
    """How the downstream model was chosen, recorded on every row so a
    defaults run and a tuned run of the same arm never get pooled.

    The fold count belongs here too. It is part of the selection procedure,
    and logging_utils keys a run on (encoder, augmentation, target, seed,
    search) with no column for it, so a 10-fold run would otherwise land on
    the same key as a 5-fold one and silently replace it on read.

    The protocol's own N_CV_SPLITS stays unsuffixed, so every row already
    logged keeps the label it was written with and is still superseded by a
    re-run of the same thing.
    """
    label = "defaults" if not cfg.n_search_iter else f"random{cfg.n_search_iter}"
    if cfg.n_cv_splits != config.N_CV_SPLITS:
        label = f"{label}-cv{cfg.n_cv_splits}"
    return label


def arm_models(cfg: config.TrainingConfig, e5_models=None, question_level: bool = False) -> dict:
    """Which checkpoint each arm's features come from.

    Logged per row so a result records the model that produced it. The run
    configuration alone cannot: it holds a single sentence_transformer_model
    while a grid run embeds with several, so every e5 arm would otherwise
    claim the same checkpoint.
    """
    suffix = "_QUESTION" if question_level else ""
    models = {f"E0_TFIDF{suffix}": "tfidf", f"E1_BETO{suffix}": cfg.beto_model}
    models.update({arm + suffix: lexical_model_tag(FEATURE_SET_OF_ARM[arm + suffix])
                   for arm in (LEXICAL_ARM, AFFECT_ARM, LEXICAL_AFFECT_ARM)})
    for arm, base in TFIDF_BASE_ARM.items():
        if arm.endswith("_QUESTION") == bool(suffix):
            models[arm] = (f"{lexical_model_tag(FEATURE_SET_OF_ARM[base])}"
                           f"+tfidf{LEXICAL_TFIDF_NGRAM}")
    for model in e5_models_of(cfg, e5_models):
        models[e5_arm(model, question_level)] = model
    models.update(mean_pooled_models(cfg, question_level))
    return models


def encoders_for(cfg: config.TrainingConfig, e5_models=None) -> list[str]:
    """Participant-level arms: TF-IDF, BETO, one per e5 checkpoint, the
    externally domain-adapted Spanish model (E7), and the three named-feature
    arms (E10 lexical, E11 affective, E12 both, E13 lexical with TF-IDF)."""
    return ["E0_TFIDF", "E1_BETO"] + [e5_arm(m) for m in e5_models_of(cfg, e5_models)] \
        + list(mean_pooled_models(cfg)) \
        + [LEXICAL_ARM, AFFECT_ARM, LEXICAL_AFFECT_ARM,
           LEXICAL_TFIDF_ARM, AFFECT_TFIDF_ARM, LEXICAL_AFFECT_TFIDF_ARM]


def question_encoders_for(cfg: config.TrainingConfig, e5_models=None) -> list[str]:
    """The same encoders trained on one row per answered question instead of
    one averaged row per participant. Evaluation stays at participant level in
    both families, so their rows share a log and compare directly.
    """
    return ["E0_TFIDF_QUESTION", "E1_BETO_QUESTION"] + \
        [e5_arm(m, question_level=True) for m in e5_models_of(cfg, e5_models)] + \
        list(mean_pooled_models(cfg, question_level=True)) + \
        [LEXICAL_ARM_QUESTION, AFFECT_ARM_QUESTION, LEXICAL_AFFECT_ARM_QUESTION,
         LEXICAL_TFIDF_ARM_QUESTION, AFFECT_TFIDF_ARM_QUESTION,
         LEXICAL_AFFECT_TFIDF_ARM_QUESTION]


def build_train_frame(df, train_ids, test_ids, cfg: config.TrainingConfig):
    """Returns (train_frame, groups). groups is None for plain StratifiedKFold,
    or an array of original-participant ids for StratifiedGroupKFold when
    augmented rows (one LLM paraphrase per training participant) are present.
    """
    if cfg.augmentation == config.AUGMENTATION_NONE:
        return df.loc[train_ids], None

    train_frame = data.load_augmented_dataset()
    splits.assert_no_leakage(test_ids, train_frame, "original_participant")
    return train_frame, train_frame["original_participant"].to_numpy()


def compute_beto_embeddings(frame, device, cfg: config.TrainingConfig,
                             model_name: str = config.BETO_MODEL):
    tokenizer, model = encoders.load_beto(device, model_name)
    return np.vstack([
        encoders.beto_embed_participant(row, tokenizer, model, device,
                                         max_length=cfg.max_tokens, stride=cfg.chunk_stride)
        for _, row in frame.iterrows()
    ])


def compute_e5_embeddings(frame, device, cfg: config.TrainingConfig, model_name: str):
    st_model = encoders.load_sentence_transformer(model_name, device)
    return np.vstack([
        encoders.st_embed_participant(row, st_model, device,
                                       max_length=cfg.max_tokens, stride=cfg.chunk_stride)
        for _, row in frame.iterrows()
    ])


def load_or_compute(path: Path, compute_fn: Callable[[], np.ndarray], ids,
                     signature: str | None = None) -> np.ndarray:
    """Disk cache for a frozen embedding matrix, keyed by file path.

    Two guards, because a stale cache is silent and would corrupt every
    result computed from it:

    - the row id order is stored and must still match, so rows can never be
      misaligned with their labels;
    - the model that produced the vectors is stored and must still match, so
      changing the checkpoint cannot quietly reuse the previous model's
      embeddings under the new name.

    A cache written before signatures existed has none recorded; it is
    stamped on first reuse and checked from then on.
    """
    ids = list(ids)
    ids_path = path.with_suffix(".ids.txt")
    meta_path = path.with_suffix(".meta.json")

    if path.exists() and ids_path.exists():
        cached_ids = ids_path.read_text(encoding="utf-8").splitlines()
        if cached_ids != ids:
            raise AssertionError(
                f"Cached embedding order at {path} does not match current row order")
        if signature is not None:
            if meta_path.exists():
                cached_signature = json.loads(meta_path.read_text(encoding="utf-8")).get("signature")
                if cached_signature != signature:
                    raise AssertionError(
                        f"Cache at {path} was produced by {cached_signature!r}, "
                        f"not {signature!r}. Delete it or use a different cache name.")
            else:
                meta_path.write_text(json.dumps({"signature": signature}), encoding="utf-8")
        return np.load(path)

    X = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    ids_path.write_text("\n".join(ids), encoding="utf-8")
    if signature is not None:
        meta_path.write_text(json.dumps({"signature": signature}), encoding="utf-8")
    return X


def compute_frozen_embeddings(frame, device, cfg: config.TrainingConfig, run_dir,
                               e5_models=None) -> dict:
    """Embedding matrix per frozen encoder, computed once and cached under
    run_dir/embeddings/. Keyed by arm id, aligned to frame.index.

    e5_models lists the sentence-transformer checkpoints to embed with; each
    becomes its own arm. Defaults to the single one in cfg.
    """
    embed_dir = run_dir / "embeddings"
    specs = [("E1_BETO", "beto.npy", cfg.beto_model,
              lambda: compute_beto_embeddings(frame, device, cfg))]
    for model in e5_models_of(cfg, e5_models):
        specs.append((e5_arm(model), _e5_cache_file(model, question_level=False), model,
                       lambda m=model: compute_e5_embeddings(frame, device, cfg, m)))
    for arm, model in mean_pooled_models(cfg).items():
        specs.append((arm, f"{model_tag(model)}.npy", model,
                       lambda m=model: compute_beto_embeddings(frame, device, cfg, model_name=m)))
    return {
        arm: load_or_compute(embed_dir / filename, compute, frame.index, signature=model)
        for arm, filename, model, compute in specs
    }


def compute_lexical_features(frame, run_dir, question_level: bool = False,
                             arms=None) -> dict:
    """{arm id: feature matrix} for the named-feature arms, cached under
    run_dir/embeddings/.

    Returned as DataFrames rather than bare arrays, unlike every other arm
    here, because these columns have names and the names are the point: an
    arm whose coefficients cannot be attributed to a feature would have no
    advantage over the embeddings it is being compared against. The cache
    stores the array and the names are rebuilt from the block registry, so
    they cannot drift from what the extractor currently produces.

    Each arm's signature is its feature-set version, so a matrix computed
    before a feature was added is rejected instead of being silently reused
    under the new name. The three sets share no cache file for the same
    reason: E12's columns are a superset of E10's, and one file for both
    would make a stale read look like a valid one.

    `arms` selects which of the three to compute; by default all of them at
    the requested granularity.
    """
    granularity = (config.GRANULARITY_QUESTION if question_level
                   else config.GRANULARITY_PARTICIPANT)
    if arms is None:
        arms = ([LEXICAL_ARM_QUESTION, AFFECT_ARM_QUESTION, LEXICAL_AFFECT_ARM_QUESTION]
                if question_level else [LEXICAL_ARM, AFFECT_ARM, LEXICAL_AFFECT_ARM])
    compute = (lexical_features.extract_question_frame if question_level
               else lexical_features.extract_frame)

    matrices = {}
    for arm in arms:
        name = FEATURE_SET_OF_ARM[arm]
        filename = f"{name.replace('-', '_')}{'_questions' if question_level else ''}.npy"
        matrix = load_or_compute(
            run_dir / "embeddings" / filename,
            lambda n=name: compute(frame, name=n).to_numpy(dtype=np.float64),
            frame.index, signature=lexical_model_tag(name))
        matrices[arm] = pd.DataFrame(
            matrix, index=frame.index,
            columns=list(lexical_features.feature_names(granularity, name)))
    return matrices


def compute_checkpoint_embeddings(frame, device, cfg: config.TrainingConfig, run_dir,
                                   checkpoints: dict, family: str = "e5") -> dict:
    """Embeddings for an explicit {arm id: checkpoint path} mapping.

    Used for locally adapted encoders, where the arm id is chosen rather than
    derived: the cache is named after the arm, since several checkpoints of
    one adaptation run share a directory name (ep05, ep10, ...) and would
    otherwise collide across runs.

    `family` picks the embedding path the checkpoint was adapted from, so an
    adapted encoder is embedded exactly like its unadapted arm: "e5" as E2
    (prefix, model pooling, normalization), "beto" as E1 (mean pooling over
    the last hidden state).
    """
    compute = {"e5": compute_e5_embeddings, "beto": compute_beto_embeddings}[family]
    embed_dir = run_dir / "embeddings"
    return {
        arm: load_or_compute(
            embed_dir / f"{arm.lower()}.npy",
            lambda p=path: compute(frame, device, cfg, model_name=str(p)),
            frame.index,
            signature=str(path),
        )
        for arm, path in checkpoints.items()
    }


def expand_to_questions(frame, group_col: str | None = None,
                         questions: list[str] = config.QUESTIONS) -> pd.DataFrame:
    """One row per answered question instead of one per participant.

    Each row carries its participant's labels, so the same target appears
    three or four times, and `group` holds the participant the row belongs
    to (the original participant when the frame is augmented). Grouping is
    what keeps every answer of one person inside the same cross-validation
    fold: without it, question 1 of a participant in training and question 3
    of the same participant in validation would leak the label.
    """
    phq9, gad7 = config.TARGET_COLUMNS["PHQ9"], config.TARGET_COLUMNS["GAD7"]
    records = []
    for participant, row in frame.iterrows():
        group = row[group_col] if group_col else participant
        for question in questions:
            text = row[question]
            if not isinstance(text, str) or not text.strip():
                continue
            records.append({
                "instance": f"{participant}__{question}",
                "participant": participant,
                "question": question,
                "group": group,
                "text": text,
                phq9: row[phq9],
                gad7: row[gad7],
            })
    return pd.DataFrame(records).set_index("instance")


def question_documents(frame, cfg: config.TrainingConfig) -> pd.Series:
    """TF-IDF input for question-level rows: one answer per document."""
    if not (cfg.tfidf_lemmatize or cfg.tfidf_normalize_numbers):
        return frame["text"]
    return frame["text"].apply(lambda t: encoders.preprocess_for_tfidf(
        t, lemmatize_text=cfg.tfidf_lemmatize, normalize_numbers_text=cfg.tfidf_normalize_numbers))


def compute_frozen_question_embeddings(frame, device, cfg: config.TrainingConfig,
                                        run_dir, e5_models=None) -> dict:
    """Embedding matrix per frozen encoder for question-level rows: each
    answer embedded on its own, with no averaging across a participant's
    questions.
    """
    def beto(model_name: str = config.BETO_MODEL):
        tokenizer, model = encoders.load_beto(device, model_name)
        return np.vstack([
            encoders.beto_embed_text(text, tokenizer, model, device,
                                      max_length=cfg.max_tokens, stride=cfg.chunk_stride)
            for text in frame["text"]
        ])

    def e5(model_name):
        model = encoders.load_sentence_transformer(model_name, device)
        return np.vstack([
            encoders.st_embed_text(text, model, device,
                                    max_length=cfg.max_tokens, stride=cfg.chunk_stride)
            for text in frame["text"]
        ])

    embed_dir = run_dir / "embeddings"
    specs = [("E1_BETO_QUESTION", "beto_questions.npy", cfg.beto_model,
              lambda: beto(cfg.beto_model))]
    for model in e5_models_of(cfg, e5_models):
        specs.append((e5_arm(model, question_level=True),
                       _e5_cache_file(model, question_level=True), model,
                       lambda m=model: e5(m)))
    for arm, model in mean_pooled_models(cfg, question_level=True).items():
        specs.append((arm, f"{model_tag(model)}_questions.npy", model,
                       lambda m=model: beto(m)))
    return {
        arm: load_or_compute(embed_dir / filename, compute, frame.index, signature=model)
        for arm, filename, model, compute in specs
    }


def lexical_tfidf_data(docs_all, lexical: pd.DataFrame, cfg: config.TrainingConfig,
                        ngram_range=LEXICAL_TFIDF_NGRAM):
    """The E13 matrix and its preprocessor: raw text in one column beside the
    named lexical features.

    A ColumnTransformer rather than a precomputed union, so the vocabulary is
    still fitted inside each cross-validation fold exactly as it is for E0.
    Fitting it once beforehand would let every fold's vectorizer see the
    words of the fold it is scored on.
    """
    frame = lexical.copy()
    frame.insert(0, "__text__", docs_all.reindex(lexical.index))
    preprocessor = ColumnTransformer([
        ("tfidf", encoders.build_vectorizer(cfg, ngram_range=ngram_range), "__text__"),
        ("lexical", StandardScaler(), list(lexical.columns)),
    ])
    return frame, preprocessor


def encoder_data(name, base_df, docs_all, embeddings: dict, cfg: config.TrainingConfig):
    if name.startswith("E0_TFIDF"):
        return docs_all, encoders.build_vectorizer(cfg)
    if name in TFIDF_BASE_ARM:
        return lexical_tfidf_data(docs_all, embeddings[TFIDF_BASE_ARM[name]], cfg)
    if name in embeddings:
        matrix = embeddings[name]
        # Already a frame for the lexical arm, whose columns are named. Kept
        # as it is so the names reach the fitted estimator; the dense arms
        # emit bare arrays and get positional columns as before.
        if isinstance(matrix, pd.DataFrame):
            return matrix, StandardScaler()
        return pd.DataFrame(matrix, index=base_df.index), StandardScaler()
    raise ValueError(name)


def aggregate_predictions(y_pred, y_true, groups):
    """Averages question-level predictions back to one score per participant.

    Evaluation stays at participant level whatever the training granularity,
    so every arm is scored on the same 48 held-out people and comparisons
    remain paired (PROTOCOL.md, Section 10). y_true is aggregated the same
    way; since every row of a participant carries that participant's label,
    its mean is just that label. The participant ids are returned alongside
    so predictions can be persisted against the person they belong to.
    """
    frame = pd.DataFrame({"group": np.asarray(groups), "pred": y_pred, "true": y_true})
    aggregated = frame.groupby("group", sort=True).mean()
    return (aggregated["pred"].to_numpy(), aggregated["true"].to_numpy(),
             aggregated.index.to_numpy())


def run_encoder(encoder_name, X_train_enc, X_test_enc, preprocessor,
                 train_frame, test_frame, groups, cfg: config.TrainingConfig,
                 test_groups=None, encoder_model=None, notebook="", verbose=True) -> list[dict]:
    """Runs cfg.seeds x TARGETS searches for one encoder, logging every
    candidate regressor tried (log_trials) and the selected model's test
    metrics (log_run). Returns the logged run rows for immediate reporting.

    test_groups switches evaluation to question-level training: when given,
    predictions are averaged within each group (participant) before any
    metric is computed.
    """
    # Question-level training is exactly the case that aggregates predictions.
    granularity = (config.GRANULARITY_QUESTION if test_groups is not None
                    else config.GRANULARITY_PARTICIPANT)
    provenance = {"family": family_of(encoder_name), "granularity": granularity,
                   "notebook": notebook}

    run_rows = []
    for seed in cfg.seeds:
        strat_key = splits.stratify_key(train_frame)
        cv = splits.cv_splitter(groups=groups, n_splits=cfg.n_cv_splits, random_state=seed)
        cv_folds = list(cv.split(X_train_enc, strat_key, groups=groups))

        for target in TARGETS:
            col = config.TARGET_COLUMNS[target]
            y_train = train_frame[col].to_numpy()
            y_test = test_frame[col].to_numpy()

            t0 = time.time()
            search_results = search_best_model(X_train_enc, y_train, preprocessor, cv_folds,
                                                seed=seed, n_iter=cfg.n_search_iter,
                                                names=cfg.regressors)
            best_name, best_estimator = best_of(search_results)

            trial_rows = [
                {
                    "encoder": encoder_name, "augmentation": cfg.augmentation, "target": target,
                    "seed": seed, "model_name": name,
                    "cv_rmse": -search.best_score_,
                    "is_selected": name == best_name,
                    "encoder_model": checkpoint_label(encoder_model) if encoder_model else "",
                    "search": search_label(cfg), **provenance,
                }
                for name, search in search_results.items()
            ]
            logging_utils.log_trials(trial_rows, cfg)

            y_pred = best_estimator.predict(X_test_enc)
            participants = test_frame.index.to_numpy()
            if test_groups is not None:
                y_pred, y_test, participants = aggregate_predictions(
                    y_pred, y_test, test_groups)
            # Clipped after aggregation so the clinical range constrains the
            # final participant-level score, exactly as in the participant-level arms.
            y_pred = metrics.clip_to_valid_range(y_pred, target)

            reg_metrics = metrics.regression_metrics(y_test, y_pred)
            clin_metrics = metrics.clinical_utility_metrics(y_test, y_pred)

            row = {
                "encoder": encoder_name, "augmentation": cfg.augmentation, "target": target,
                "seed": seed, "fold": "test", "split": "confirmatory",
                **reg_metrics,
                "band_f1_macro": clin_metrics["band_f1_macro"],
                "band_qwk": clin_metrics["band_qwk"],
                "band_confusion_matrix": clin_metrics["band_confusion_matrix"],
                "screening_f1": clin_metrics["screening_f1"],
                "screening_auc": clin_metrics.get("screening_auc", ""),
                "screening_confusion_matrix": clin_metrics["screening_confusion_matrix"],
                "selected_model": best_name,
                "selected_params": search_results[best_name].best_params_,
                "encoder_model": checkpoint_label(encoder_model) if encoder_model else "",
                "search": search_label(cfg), **provenance,
                "notes": "",
            }
            logging_utils.log_run(row, cfg)
            logging_utils.log_predictions(row, participants, y_test, y_pred)
            run_rows.append(row)
            if verbose:
                print(f"{encoder_name} seed={seed} target={target} best={best_name} "
                      f"rmse={reg_metrics['rmse']:.3f} r2={reg_metrics['r2']:.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return run_rows
