"""Central configuration for the text-channel regression experiment.

Two kinds of settings live here, deliberately kept apart:

- Protocol-frozen constants (the hold-out split, its seed, the clinical
  cutoffs, the target definitions). Changing any of these breaks
  comparability with every result already recorded under this protocol, so
  they are module constants, not knobs.
- TrainingConfig: the tunable part of a run (seeds, search budget, encoder
  choices, TF-IDF preprocessing). An instance is created at the entry point
  (notebook or script), passed down explicitly, and logged with every result
  row, so a run is fully described by its configuration.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------
# text_experiment/ lives directly inside the main repository, which holds
# the shared data/ directory. Deriving the path from __file__ keeps it
# independent of the notebook's working directory.
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
DATA_DIR = REPO_ROOT / "data"


def require_data_dir(path: Path = None) -> Path:
    """The shared data directory, or a message saying what is missing.

    Checked here rather than at import: the interview recordings and scores
    are not part of this repository and never will be, so a clone has no
    data/ beside it. Raising on import would make the package unimportable,
    and with it the tests, the diagnostics and anything else that does not
    touch a participant's text.
    """
    path = Path(path or DATA_DIR)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected the shared data directory at {path}. It holds one "
            f"folder per session (UTB_*) with the transcriptions and the "
            f"PHQ-9/GAD-7 scores, and is not distributed with this code.")
    return path

# External corpus used only as unlabeled text for TSDAE adaptation. It sits
# beside the repository rather than inside data/, and is checked when loaded,
# not here, since nothing but the adaptation step needs it.
MENTALRISKES_DIR = REPO_ROOT.parent / "MentalRiskES-Corpus" / "data"

RESULTS_DIR = EXPERIMENT_ROOT / "results"
EXPORTS_DIR = RESULTS_DIR / "exports"
MODELS_DIR = EXPERIMENT_ROOT / "models"

# Experiment families and input granularities. Results are laid out as
# results/<family>/<augmentation>/<granularity>/, because each combination
# trains on different rows and therefore needs its own embedding cache and
# its own log; what you want side by side instead lives in columns.
FAMILY_FROZEN = "frozen"
FAMILY_FINETUNING = "fine-tuning"
# Fine-tuning with an inner cross-validation instead of a single holdout.
# A separate family, not a separate arm name, so the two schemes appear as
# the same encoder measured two ways and can be put side by side.
FAMILY_FINETUNING_CV = "fine-tuning-cv"
# One encoder with two regression heads, trained on PHQ-9 and GAD-7 at once
# (src/multitask.py). A separate family, not a separate arm name, because a
# row here is one of two read-outs of a single training run rather than a run
# of its own, and pooling it with the single-target rows would double-count
# that run.
FAMILY_FINETUNING_MT = "fine-tuning-multitask"
# Named features -- lexical categories, Spanish affective norms, TF-IDF --
# rather than a pretrained encoder's dimensions (src/lexical_features.py).
# Procedurally these are frozen arms: the representation is computed once and
# a classical regressor is fitted on top. They are kept as their own family
# so the named-feature study reads as one thing in the results tree instead
# of being interleaved with the encoder comparison it sits beside. Both are
# still scored on the same held-out participants, so comparing across the two
# remains possible; it just has to be asked for rather than falling out of
# the layout.
FAMILY_LEXICAL = "lexical"
GRANULARITY_PARTICIPANT = "participant"
GRANULARITY_QUESTION = "question"

AUGMENTATION_NONE = "no_augmentation"
AUGMENTATION_LLM = "augmented"


def results_dir(augmentation: str, family: str = FAMILY_FROZEN,
                 granularity: str = GRANULARITY_PARTICIPANT) -> Path:
    """Where one combination's log and embedding cache live.

    Three levels, each because the combination changes what is on disk:
    augmented runs train on different rows than plain ones, question-level
    runs embed answers rather than participants, and fine-tuning produces no
    embedding cache at all. Mixing any of them in one directory is how a
    stale cache or a superseded row goes unnoticed.
    """
    return RESULTS_DIR / family / augmentation / granularity

# --- Protocol-frozen: the hold-out split and its seed -------------------
# Every arm is evaluated on the same held-out participants so comparisons
# stay paired (PROTOCOL.md, Sections 8 and 10). Do not tune these.
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_BOOTSTRAP = 2000

# Defaults for the tunable settings below; see TrainingConfig.
# One seed. Every arm is scored on the same fixed test participants, so the
# seed does not change who is evaluated, only the stochastic parts of
# fitting: the cross-validation fold assignment, a regressor's own
# randomness, and, for the fine-tuned arms, the weight initialization of the
# head and the validation slice. Running several of those and reporting the
# spread was how this study estimated its own measurement noise; with one
# seed that estimate is gone, and a difference between arms can no longer be
# read against it (PROTOCOL.md, Sections 10 and 13). The tuple is kept so a
# variance estimate can be restored by listing more seeds here.
SEEDS = (42,)
N_CV_SPLITS = 5

# 0 means no hyperparameter search: every candidate regressor is fitted with
# its library defaults and cross-validation still picks the best family. With
# ~88 samples per fold the CV estimate is noisy enough that maximizing over
# many draws mostly selects fold noise, so a broad search buys little while
# costing ~50x the compute (2250 fits per arm against 45).
N_SEARCH_ITER = 0

# Budget for tuning one already-chosen model family, which is a much smaller
# problem than searching all nine.
SELECTIVE_TUNING_ITER = 15

# --- Data -----------------------------------------------------------------
QUESTIONS = ["Q01", "Q02", "Q03", "Q04"]
QUESTIONS_REQUIRED = ["Q01", "Q02", "Q03"]

# --- Protocol-frozen: targets and clinical cutoffs ----------------------
PRIMARY_TARGET = "PHQ9"
SECONDARY_TARGET = "GAD7"
TARGET_COLUMNS = {"PHQ9": "PHQ9_Total", "GAD7": "GAD7_Total"}
SCORE_RANGE = {"PHQ9": (0, 27), "GAD7": (0, 21)}
BINARY_THRESHOLD = 10  # standard PHQ-9 / GAD-7 screening-positive cutoff

# --- Encoders -----------------------------------------------------------
BETO_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
# E2 off-the-shelf, and the checkpoint TSDAE adapts in Phase C (E3/E5). One
# backbone for both so E3 vs E2 isolates domain adaptation rather than
# confounding it with model size. The exploratory phase used the -large
# variant (1024-dim); -small emits 384-dim embeddings, which matters at
# n=110 training participants, and is the least prone to overfitting the
# small in-domain corpus the adaptation step has to work with.
SENTENCE_TRANSFORMER_MODEL = "intfloat/multilingual-e5-small"

# E7: a Spanish RoBERTa already adapted to the mental-health domain by others
# (ELiRF, Universitat Politècnica de València): roberta-large-bne continued
# with masked-language-model pretraining on ~1.9M mental-health Reddit posts
# machine-translated into Spanish. It is what this study cannot build itself,
# a domain-adaptive pretraining at scale, so it is evaluated as its own arm.
# Embedded exactly like BETO (mean pooling over the last hidden state), since
# it is a plain masked-language model and not a sentence-transformer.
# Its own base checkpoint is no longer published, so a gain cannot be split
# between the domain adaptation and the backbone (PROTOCOL.md, Section 7).
MENTAL_ES_MODEL = "ELiRF/RoBERTa-es-mental-large"
MAX_TOKENS = 512
CHUNK_STRIDE = 256
# e5 was trained with this prefix on every input. It lives here rather than in
# encoders.py because TSDAE adaptation must use exactly the same one: adapting
# on unprefixed text and then embedding with the prefix evaluates the encoder
# on an input distribution it was never adapted to.
E5_PREFIX = "query: "


# --- Tunable settings ---------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """The tunable part of one run, logged alongside its results.

    `n_search_iter=0` fits each candidate with library defaults and lets
    cross-validation choose the family, which is a declared deviation from
    the exploratory phase's 50-draw random search (PROTOCOL.md, Section 7).
    Rows record which regime produced them, so the two are reported side by
    side rather than silently pooled.
    """

    augmentation: str = AUGMENTATION_NONE
    seeds: tuple[int, ...] = SEEDS
    n_cv_splits: int = N_CV_SPLITS
    n_search_iter: int = N_SEARCH_ITER

    # Which candidate regressors cross-validation chooses between; empty means
    # all of them (src/regression_models.py). Narrowing this is a change to the
    # selection procedure rather than a speed-up: the winner is drawn from a
    # smaller pool, so a row produced under a restricted set does not compare
    # with one produced under the full set. It belongs to the configuration
    # for that reason, and is therefore logged with every row it produces.
    regressors: tuple[str, ...] = ()

    beto_model: str = BETO_MODEL
    sentence_transformer_model: str = SENTENCE_TRANSFORMER_MODEL
    # Set to "" to drop the E7 arm, e.g. when the checkpoint is not available.
    mental_es_model: str = MENTAL_ES_MODEL
    max_tokens: int = MAX_TOKENS
    chunk_stride: int = CHUNK_STRIDE

    # TF-IDF only (E0): BETO/e5 are subword-tokenized models pretrained on
    # natural cased text and take the raw transcription instead.
    tfidf_max_features: int = 300
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 3
    tfidf_max_df: float = 0.90
    tfidf_lemmatize: bool = True
    tfidf_normalize_numbers: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FineTuningConfig:
    """End-to-end fine-tuning of an encoder with a regression head.

    Unlike the frozen arms, where the encoder never learns anything about
    the task, here the prediction error backpropagates through the whole
    transformer, so the representation reorganizes for this target. That is
    the only approach in the study that optimizes the actual objective
    rather than a proxy, and also the one most exposed to overfitting: a few
    hundred training rows against hundreds of millions of parameters.

    The defaults carry the mitigations the exploratory phase used: layer-wise
    learning-rate decay so the lower, more general layers move least, warmup,
    and early stopping on a held-out slice of the training participants.
    """

    augmentation: str = AUGMENTATION_NONE
    seeds: tuple[int, ...] = SEEDS
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum_steps: int = 1
    max_tokens: int = MAX_TOKENS

    lr_head: float = 1e-4
    lr_encoder_top: float = 3e-5
    llrd_decay: float = 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0

    # Validation slice for early stopping, split by participant so no one's
    # answers appear on both sides.
    val_fraction: float = 0.2
    use_bf16: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FineTuningCVConfig(FineTuningConfig):
    """Fine-tuning with an inner cross-validation over the training
    participants, which is what notebooks 01-03 do for their regressors and
    what the exploratory phase did before Section 9b replaced it with a
    single holdout.

    The folds decide one thing only: how many epochs to train. Each fold
    early-stops on its own validation part, the chosen budgets are pooled,
    and the model that is actually evaluated is refitted on all of the
    training rows for that many epochs with no early stopping. Two things
    are bought with the extra compute. Every training participant reaches
    the final model, instead of 20 percent being held back; and the epoch
    budget rests on the whole training set rather than on one slice of 22
    people, which Section 13 records as an unreliable signal.

    The cost is k+1 trainings per arm and target instead of one.
    """

    n_cv_splits: int = N_CV_SPLITS

    def as_dict(self) -> dict:
        return asdict(self)


# Text sources for TSDAE adaptation. The interviews are always included: the
# declared variants are interviews alone, and interviews plus MentalRiskES.
CORPUS_INTERVIEWS = "interviews"
CORPUS_MENTALRISKES = "mentalriskes"
TSDAE_CORPORA = {
    (CORPUS_INTERVIEWS,): "tsdae_e5_small_v2",
    (CORPUS_INTERVIEWS, CORPUS_MENTALRISKES): "tsdae_e5_small_v2_mentalriskes",
}


# Checkpoints this study adapts with MLM, by the short name the CLI takes.
# BETO is the Spanish backbone whose frozen and fine-tuned arms are already
# measured; the ELiRF model arrives already domain-adapted, so adapting it
# further is the DAPT-then-TAPT combination (PROTOCOL.md, Section 9d).
MLM_BACKBONES = {"beto": BETO_MODEL, "robertaes-mental": MENTAL_ES_MODEL}
MLM_OUTPUT_PREFIX = {BETO_MODEL: "mlm_beto", MENTAL_ES_MODEL: "mlm_robertaes_mental"}


def mlm_output_name(corpora: tuple[str, ...], lexicon_masking: bool,
                    base_model: str = BETO_MODEL) -> str:
    """Directory under models/ for one MLM adaptation variant. Backbone,
    corpus and masking scheme are all part of the name, so no two variants
    can overwrite each other."""
    name = MLM_OUTPUT_PREFIX.get(base_model, "mlm_" + base_model.rsplit("/", 1)[-1].lower())
    if CORPUS_MENTALRISKES in corpora:
        name += "_mentalriskes"
    if lexicon_masking:
        name += "_lexicon"
    return name


@dataclass(frozen=True)
class MLMConfig:
    """Task-adaptive pretraining (TAPT) of BETO: masked-language-model
    training continued on this study's own unlabeled text.

    TAPT rather than domain-adaptive pretraining because the corpus is the
    task's own text (~100k tokens), not a large domain collection. Defaults
    follow BETO's pretraining where it matters for continuity (whole-word
    masking at 15 percent, 80/10/10 replacement) and the TAPT recipe for small
    corpora (many epochs), with a lower learning rate than TAPT's 1e-4 to limit
    forgetting of general Spanish on so little text.

    Examples are whole documents, not sentences: an interview answer, or
    consecutive turns of one MentalRiskES session, up to max_tokens.

    Masking is applied per batch, so every epoch sees different masks. With
    lexicon_masking, words of the clinical lexicon (src/lexicon.py) are masked
    with lexicon_mask_probability and every other word at the rate that keeps
    the overall share at mask_probability.
    """

    base_model: str = BETO_MODEL
    corpora: tuple[str, ...] = (CORPUS_INTERVIEWS,)
    max_tokens: int = MAX_TOKENS

    mask_probability: float = 0.15
    lexicon_masking: bool = False
    lexicon_mask_probability: float = 0.40

    # Overrides the checkpoint's own dropout while adapting. None keeps it.
    # The MarIA-derived checkpoints ship with dropout 0.0, which is a
    # reasonable setting for a 570GB pretraining corpus and a poor one for
    # 400 documents, so those runs pass 0.1 here.
    dropout: float | None = None

    epochs: int = 100
    # Declared before training; every one is saved and evaluated downstream.
    checkpoint_epochs: tuple[int, ...] = (10, 25, 50, 100)
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    use_fp16: bool = True

    # Held out from training to monitor MLM loss, grouped by person: a share
    # of the training participants' interviews and whole MentalRiskES
    # sessions. Never the study's test participants, which never enter at all.
    val_fraction: float = 0.10
    val_mentalriskes_sessions: int = 1

    seed: int = RANDOM_STATE
    output_name: str = "mlm_beto"

    def __post_init__(self):
        if tuple(self.corpora) not in TSDAE_CORPORA:
            raise ValueError(f"Undeclared corpus combination {self.corpora}; "
                             f"expected one of {list(TSDAE_CORPORA)}")
        if max(self.checkpoint_epochs) > self.epochs:
            raise ValueError(f"checkpoint_epochs {self.checkpoint_epochs} exceed epochs={self.epochs}")
        if self.lexicon_masking and self.lexicon_mask_probability <= self.mask_probability:
            raise ValueError("lexicon_mask_probability must exceed mask_probability to mean anything")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TSDAEConfig:
    """Unsupervised domain adaptation of the sentence-transformer (Phase C,
    step C1). Defaults follow the TSDAE paper's recipe, which is what has
    been validated for domain adaptation on small corpora: 60 percent word
    deletion, a tied encoder-decoder, a constant learning rate with no
    warmup, and no weight decay.

    TSDAE's reconstruction loss measures how well the decoder rebuilds text,
    not how useful the embeddings are, so it gives no trustworthy stopping
    signal. Rather than betting on one epoch budget, the run saves periodic
    checkpoints and every declared one is evaluated and reported. That makes
    adaptation length a described curve rather than a tuned choice: picking
    the best-scoring checkpoint on the held-out set would be selecting the
    encoder on the outcome it is meant to be evaluated against.
    """

    base_model: str = SENTENCE_TRANSFORMER_MODEL
    deletion_ratio: float = 0.6
    epochs: int = 30
    # Intermediate checkpoints, so downstream performance can be read as a
    # curve over adaptation length instead of resting on one fixed budget.
    # Saving is periodic because the training API takes a step interval, not
    # a list of epochs; which of them get evaluated is declared separately.
    checkpoint_every_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    scheduler: str = "constant"   # transformers' lr_scheduler_type name
    tie_encoder_decoder: bool = True
    min_sentence_words: int = 5
    # Whisper does not always punctuate, so the segmenter occasionally emits a
    # whole answer as one "sentence". Those exceed the encoder's context and
    # would be truncated, leaving the decoder reconstructing a tail it never
    # saw, so they are split into consecutive pieces rather than dropped: at
    # ~2000 training sentences, discarding a long passage is expensive.
    max_sentence_words: int = 60
    # Prepended to the corrupted encoder input only, after deletion so it is
    # never deleted itself; the decoder still reconstructs the bare sentence.
    # Must match the prefix the downstream embedding uses (encoders.py).
    encoder_prefix: str = E5_PREFIX
    # The e5 pipeline ends in a Normalize module. TSDAE feeds the pooled
    # vector to the decoder's cross-attention, and a unit-norm vector is an
    # order of magnitude smaller than the hidden states that attention was
    # built for, so the decoder is given the unnormalized pooled vector. The
    # saved checkpoints keep Normalize, so downstream use is unaffected.
    normalize_during_training: bool = False
    # Which texts the adaptation corpus is built from; a key of TSDAE_CORPORA.
    corpora: tuple[str, ...] = (CORPUS_INTERVIEWS,)
    seed: int = RANDOM_STATE
    # "_v2": the "tsdae_e5_small" checkpoints were produced by an implementation
    # that froze the deletion noise and never trained the decoder (see
    # domain_adaptation.py). Each corpus variant writes to its own directory
    # (TSDAE_CORPORA), so the two runs can never overwrite each other.
    output_name: str = TSDAE_CORPORA[(CORPUS_INTERVIEWS,)]

    def __post_init__(self):
        if tuple(self.corpora) not in TSDAE_CORPORA:
            raise ValueError(f"Undeclared corpus combination {self.corpora}; "
                             f"expected one of {list(TSDAE_CORPORA)}")

    def as_dict(self) -> dict:
        return asdict(self)
