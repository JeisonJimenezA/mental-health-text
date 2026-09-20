# Experimental Protocol: Text-Channel Encoder Comparison for PHQ-9 / GAD-7 Prediction

Status: confirmatory protocol, fixed before the domain-adaptation and
fine-tuning experiments it governs. Amended since, and each amendment says
what forced it: the design changes recorded in Sections 7, 9 and 9b were
made in response to measured outcomes, not to improve results after seeing
them. Hypotheses are left as originally written, with their outcome noted.
Scope: text channel only. Audio and video channels, and multimodal fusion,
are out of scope for this protocol.

## 1. Background

The parent project collects semi-structured interview sessions with
university students, conducted by a virtual interviewer, together with
self-reported PHQ-9 (depression) and GAD-7 (anxiety) severity scores. Prior
exploratory work in `../notebooks/04_text/` compared several text encoders
(TF-IDF, frozen BETO, a frozen multilingual sentence-transformer, and
end-to-end fine-tuned transformers) as predictors of these scores. That work
established the data pipeline, the held-out split, and leakage safeguards
that this protocol reuses unchanged.

## 2. Relationship to Prior Work

The results already obtained in `../notebooks/04_text/01` through `05` are
treated as pilot/exploratory findings that motivated this protocol, not as
part of its confirmatory evidence. Two reasons:

1. Every prior notebook was evaluated against the same fixed 48-participant
   hold-out set. Iterating model and encoder choices while repeatedly
   inspecting that same hold-out is a form of implicit overfitting to that
   specific sample, independent of any leakage at the data level.
2. Those experiments used a single training seed, so the reported
   differences between encoders carried no variance estimate. This protocol
   originally addressed that with five seeds per arm; it now runs one
   (Section 10), so the variance estimate is again absent and the
   distinction from the exploratory phase rests on point 1 alone.

Framing prior results as exploratory, and reserving the hold-out set for a
single confirmatory evaluation per encoder under this protocol, avoids
retroactively writing hypotheses to match already-known results.

## 3. Research Questions

RQ1: Does a frozen multilingual sentence-transformer produce better PHQ-9
regression performance than frozen BETO embeddings from the same session
transcriptions?

RQ2: Does adapting a sentence-transformer to the project's own vocabulary
through unsupervised domain adaptation (TSDAE) on the training transcripts,
prior to downstream use, improve regression performance relative to both
off-the-shelf BETO and the off-the-shelf sentence-transformer?

RQ3: Does end-to-end fine-tuning of a sentence-transformer outperform using
it as a frozen encoder with a classical regression head? (Originally posed
for the domain-adapted encoder; since the adaptation was measured to degrade
the representation, the question is now asked of the unadapted checkpoint —
see Section 7.)

RQ4 (secondary, exploratory): Does LLM-based paraphrase augmentation
interact with encoder choice, i.e. does the size of its benefit differ
across encoders?

## 4. Hypotheses

H1: The off-the-shelf multilingual sentence-transformer achieves lower test
RMSE on PHQ-9 than frozen BETO.

H2: The domain-adapted sentence-transformer achieves lower test RMSE on
PHQ-9 than both frozen BETO and the off-the-shelf sentence-transformer.
**Refuted** (Section 9): every adaptation length tested was worse than the
unadapted encoder, on both targets. The hypothesis is left as written rather
than revised to fit the outcome.

H3: The fine-tuned sentence-transformer achieves lower test RMSE on PHQ-9
than its frozen counterpart. (Stated originally of the domain-adapted
encoder; H2's refutation makes that starting point untenable, so it is
tested from the unadapted checkpoint.)

RQ4 has no directional hypothesis; it is reported descriptively.

## 5. Task Formulation

Regression is the only trained task. Each encoder arm produces one
continuous PHQ-9 (and, secondarily, GAD-7) prediction per participant.

Severity-band classification (four standard clinical bands) and
screening-positive classification (threshold of 10) are derived from that
same continuous prediction by applying fixed cutoffs, and are reported as
measures of clinical utility, not as separately trained models. This
reduces the design from three trained tasks per encoder to one, and avoids
diluting the primary comparison across three independent metric families.

Rationale: PHQ-9 and GAD-7 are validated as continuous severity scores;
discretizing a continuous outcome loses statistical power and introduces
arbitrary cutoffs, which is undesirable given the limited sample size
available for hypothesis testing (n=48 in the test set).

A bounded secondary analysis (Section 11) checks whether a classifier
trained directly on severity bands detects minority classes better than
thresholding the regression output, without expanding the primary design.

## 6. Primary and Secondary Endpoints

Primary endpoint: RMSE of PHQ-9 regression on the held-out test set.

R-squared is reported alongside RMSE as an interpretability measure. On a
fixed test set, R-squared equals 1 minus MSE divided by the variance of the
true PHQ-9 values in that set; since this variance is constant across
models, ranking encoders by RMSE and ranking them by R-squared always agree.
Reporting both therefore does not introduce a second, independent decision
criterion.

MAE is reported as a robustness check, since it reflects a different error
geometry (L1 rather than L2) and can occasionally disagree with the
RMSE/R-squared ranking when a model makes a small number of large errors.

Secondary/confirmatory endpoints, evaluated with the same procedure and
subject to multiple-comparison correction:
- RMSE, MAE and R-squared of GAD-7 regression.
- Severity-band F1-macro, quadratic weighted kappa, and screening-positive
  F1/AUC, derived from the PHQ-9 and GAD-7 predictions as described in
  Section 5.

## 7. Experimental Factors

Primary factor: encoder, with the following levels.

| Level | Description | Training mode |
|-------|-------------|----------------|
| E0 | TF-IDF | classical regressor on top |
| E1 | BETO, mean-pooled | frozen, classical regressor on top |
| E2 | Off-the-shelf multilingual sentence-transformer, three sizes | frozen, classical regressor on top |
| E3 | `multilingual-e5-small` domain-adapted with TSDAE, four checkpoints | frozen, classical regressor on top |
| E5 | Seven unadapted checkpoints across three model families | end-to-end fine-tuned, single validation holdout |
| E6 | The same seven checkpoints | end-to-end fine-tuned, 5-fold cross-validated epoch budget |

E3 adapts the same `-small` checkpoint that E2 uses off the shelf. Holding
the backbone constant is what makes E3 vs E2 a test of domain adaptation:
had E3 adapted a different size, a loss could not be attributed to the
adaptation rather than to the change of backbone.

**E5 starts from the unadapted checkpoint**, not from E3 as originally
specified. The change is forced by measurement, not convenience: every TSDAE
checkpoint degraded the representation relative to the unadapted encoder, at
every adaptation length tested (Section 9), so fine-tuning from one would
begin with a demonstrated handicap. E4 is folded into E5 rather than
dropped: BETO fine-tuned is one of the seven E5 checkpoints, which makes it
the single arm whose frozen and fine-tuned versions share a checkpoint
exactly.

E0, E1 and E2 were already produced in the exploratory phase and are
re-run here under this protocol's declared procedure and logging; their
encoder
and training procedure are unchanged from the exploratory notebooks, with
one addition: E0's input text is now normalized before vectorization
(digit runs replaced with a placeholder, then lemmatized with spaCy's
`es_core_news_sm`, `src/encoders.py`). This applies only to E0.
E1-E5 are subword-tokenized models pretrained on natural, cased Spanish
text; they learn casing and inflectional variation from context, so
normalizing their input first would push it outside the pretraining
distribution rather than help. E3 and E5 are new.

The exploratory phase selected each downstream regressor with a 50-draw
random search. **That search is dropped: every candidate is fitted with its
library defaults and cross-validation still chooses the family, only the
within-family tuning is skipped.** At roughly 88 samples per fold the CV
estimate is noisy enough that maximizing over many draws largely selects
fold noise, which the observed selection instability supports directly (the
cross-validation cannot reliably separate SVR from GradientBoosting; the gap
changes sign with the fold assignment). Every row records which regime
produced it, so the two are reported side by side rather than pooled, and
the comparison measures what the search was worth instead of assuming it.
Tuning one already-chosen family, when warranted, uses a 15-draw budget and
must be decided from the cross-validation ranking, never from test scores.

Within E5 the checkpoint is varied along three declared contrasts, each
stated before any of the arms was run (Section 9b): capacity, pretraining
language coverage, and architecture. The three are not independent, since a
single checkpoint contributes to more than one, but each has at least one
pair that differs in the contrasted factor alone.

Within E2 the checkpoint is varied across three sizes of the same family,
which share a tokenizer and a 512-token context and differ only in depth and
embedding width: `multilingual-e5-small` (384), `-base` (768) and `-large`
(1024). They form an ordered capacity series, so their comparison reads as
whether capacity helps monotonically or plateaus.

Secondary factor: input granularity, with two levels, crossed with the frozen
encoders E0, E1 and E2:

- participant level: a participant's per-question embeddings are averaged
  into one vector, one training row per participant;
- question level: each answered question is its own training row, carrying
  its participant's score, with cross-validation grouped by participant and
  predictions averaged back per participant before any metric is computed.

Both levels are evaluated on the same held-out participants, so they compare
directly. This crossing is not optional bookkeeping: the fine-tuned arms
(E5) operate at question level by construction, so comparing them against
participant-level frozen arms would confound method with granularity. The
question-level frozen arms are their matched control.

Encoder (five levels: TF-IDF, BETO, and the three e5 checkpoints) crossed
with granularity gives a ten-cell grid. It is run complete and reported
complete, including cells that lose. The grid is descriptive: it
characterizes capacity and granularity, and no confirmatory claim is drawn
from it. The confirmatory claims remain the pre-registered comparisons of
RQ1 to RQ3 (Section 10), and the test-set exposure the grid adds is recorded
in Section 13.

Third factor: data augmentation (LLM paraphrase augmentation on or off),
crossed only with the frozen encoders E1, E2 and E3, to bound the number of
arms.

Variables held constant across all arms: the held-out split (Section 8),
the regression loss and metric definitions, the model-selection procedure
for the classical regressors, and the mean-pooling strategy for frozen
encoders.

Unit of experimental analysis: participant. Each participant contributes
one PHQ-9 value, one GAD-7 value, and (at question level) between one and
four text segments that are aggregated back to the participant before
evaluation, matching the aggregation the exploratory fine-tuning notebook
used.

## 8. Data and Split Procedure

Dataset: 158 participants with at least one usable question transcription
and complete PHQ-9/GAD-7 totals, out of 171 participants with complete
demographic and score records. PHQ-9 mean 10.4 (SD 5.8), GAD-7 mean 8.5
(SD 5.0), both scales showing meaningful class imbalance at the severe end
(documented in the exploratory phase).

Split: fixed 70/30 train/test split over participants, stratified by a
combined PHQ9-band by GAD7-band key, seed 42. This yields 110 training and
48 test participants, unchanged from the exploratory phase (`src/splits.py`,
function `train_test_participants`).

Leakage safeguards, enforced in code before any augmented row enters
training or cross-validation:
- The test set is fixed once and reused identically by every encoder arm.
- Cross-validation during hyperparameter search uses StratifiedGroupKFold,
  grouped by original participant, whenever augmented rows are present, so
  a paraphrase and its source never fall in different folds.
- An explicit assertion (`src/splits.py`, function `assert_no_leakage`)
  raises if any augmented row traces back to a test participant. This
  guard was added after a real leakage incident was identified in the
  exploratory phase (inflated cross-validation metrics relative to
  hold-out performance).
- The domain-adaptation corpus for E3 (Section 9) is restricted to the
  110 training participants only, even though the adaptation objective
  does not use PHQ-9/GAD-7 labels, to avoid transductive exposure of test
  participants' text to the encoder before final evaluation.

## 9. Domain Adaptation Procedure (E3)

Method: TSDAE (Transformer-based Sequential Denoising Auto-Encoder), the
standard unsupervised domain-adaptation method for sentence-transformers
under a small, unlabeled corpus. A continued masked-language-modeling pass
is not used as the primary method, since the in-domain corpus (training
transcripts only, on the order of 300 to 400 short question-answer
segments) is too small to adapt a transformer from a general-domain
masked-language-modeling objective without a high risk of overfitting.

Base checkpoint: `intfloat/multilingual-e5-small`, rather than a larger
variant, to reduce the risk of overfitting the adaptation step to a small
corpus, and because its 384-dimensional output leaves the downstream
regressor a far less severe p >> n problem at n=110 training participants.
This is the same checkpoint E2 uses off-the-shelf, so E3 vs E2 measures
adaptation with the backbone held constant (Section 7).

Adaptation corpus: transcriptions from the 110 training participants only
(Section 8), never the LLM paraphrases. Paraphrases carry the generator's
register rather than the students', they exist only for minority PHQ-9
classes, and making the corpus depend on the augmentation setting would
produce two different adapted encoders and entangle adaptation with
augmentation instead of keeping them separable. Answers are split into
sentences, since the denoising objective operates at sentence level and one
document per participant would leave ~110 training examples; unpunctuated
run-ons above 60 words are split rather than discarded. The corpus is 2108
sentences.

Adaptation length: rather than betting on one epoch budget, a single
continuous 30-epoch run saves a checkpoint every 5 epochs, and the declared
four (5, 10, 20, 30) are each evaluated and reported. This supersedes the
original commitment to fix the epoch count a priori. The reconstruction loss
measures how well the decoder rebuilds text, not how useful the embeddings
are, so it gives no trustworthy stopping signal; describing adaptation
length as a curve avoids both betting blindly and selecting the
best-scoring checkpoint on the held-out set, which would be choosing the
encoder on the outcome it is meant to be evaluated against.

Intrinsic validation: before the adapted encoders enter the downstream
comparison, their embedding space is checked on training data only, through
the spread of pairwise cosine similarities (label-free, and the diagnostic
for TSDAE's known collapse failure mode), the silhouette of the PHQ-9
severity bands, and nearest-neighbour coherence. This exists to separate a
failure of adaptation from a failure of the downstream head.

Artifact management: checkpoints are saved under `models/<name>/ep<NN>/`
with an `adaptation_metadata.json` recording the corpus size and every
hyperparameter used.

**Outcome.** The adaptation did not help, and the result is recorded here
rather than set aside. Intrinsic validation at 5 epochs already showed the
space contracting (mean pairwise cosine 0.964 to 0.989) and the band
silhouette worsening (-0.023 to -0.090). Downstream, all four checkpoints
were worse than the unadapted encoder on both targets — eight comparisons
out of eight — losing 0.29 to 0.65 RMSE against seed-to-seed SDs of 0.06 to
0.49, with several checkpoints reaching R-squared at or below zero. More
adaptation did not recover: PHQ-9 was worst at 30 epochs. The most plausible
mechanism is corpus size: TSDAE's decoder begins with randomly initialized
cross-attention and language-modeling head, and 2108 sentences is one to two
orders of magnitude below the corpora where the method is validated, so what
reaches the encoder is largely noise. This answers RQ2 in the negative.

## 9b. Fine-Tuning Procedure (E5)

Every other arm keeps the encoder frozen: it emits vectors and a separate
classical model learns from them, so the transformer never learns anything
about severity. Here a regression head is attached and the prediction error
backpropagates through the whole encoder. It is the only approach in the
study that optimizes the actual objective rather than a proxy, and the one
most exposed to overfitting.

Seven checkpoints are run and all seven reported, chosen so that the set
answers three questions the frozen grid cannot.

| Arm | Checkpoint | Parameters (total / encoder) |
|-----|------------|------------------------------|
| E5_E5SMALL_FT | `intfloat/multilingual-e5-small` | 118M / 86M |
| E5_E5BASE_FT | `intfloat/multilingual-e5-base` | 278M / 86M |
| E5_E5LARGE_FT | `intfloat/multilingual-e5-large` | 560M / 303M |
| E5_BETO_FT | `dccuchile/bert-base-spanish-wwm-cased` | 110M / 86M |
| E5_BERTIN_FT | `bertin-project/bertin-roberta-base-spanish` | 125M / 86M |
| E5_MDEBERTA_FT | `microsoft/mdeberta-v3-base` | 279M / 86M |
| E5_XLMR_FT | `xlm-roberta-base` | 278M / 86M |

*Capacity.* The three e5 sizes span 118M to 560M parameters on one
pretraining recipe. Frozen, capacity helped monotonically; fine-tuned the
risk runs the other way, since more parameters have more to overfit against
a few hundred training rows. Which effect dominates is empirical, so it is
declared in advance rather than assumed.

*Pretraining language coverage.* BETO and BERTIN are pretrained on Spanish
alone; the e5 checkpoints and XLM-R on roughly a hundred languages at once.
Spanish-only pretraining spends its capacity on this language, multilingual
pretraining sees far more text: which trade wins on Spanish clinical
interview speech is the question. BETO and BERTIN also differ from each
other in pretraining objective (BERT with whole-word masking against
RoBERTa), which separates that from the language.

*Architecture.* mDeBERTa-v3 against XLM-R is a controlled pair: the same
pretraining corpus (CC100), the same 250k vocabulary and near-identical
size, differing in DeBERTa's disentangled attention with relative position
encodings. Without XLM-R in the set, any result for mDeBERTa would confound
architecture with multilingual pretraining.

Total parameter counts are not a capacity ranking across these arms. The
250k-token vocabulary of mDeBERTa, XLM-R and e5-base puts about 192M
parameters into an embedding table; their encoders are ~86M, the same as
BETO's. Only the e5 ladder varies encoder capacity.

The Spanish RoBERTa slot is filled by BERTIN rather than MarIA
(`PlanTL-GOB-ES/roberta-base-bne`), which would otherwise be the stronger
candidate: its weights are no longer published, the repository returning 404
for its configuration file, with only third-party fine-tuned forks
remaining. No Spanish-only DeBERTa with comparable standing exists, so the
DeBERTa family is represented by the multilingual v3 checkpoint.

Mitigations: layer-wise learning-rate decay, so the lower and more general
layers move least; warmup; and early stopping on a validation slice split by
participant, never by row. Learning rates are assigned by a parameter's
position in the network rather than by its name, since these architectures
do not agree on what a base model contains: BERT-family models carry a
pooler above the encoder, DeBERTa keeps relative-position embeddings and a
normalization layer on the encoder itself. Anything shared across layers is
grouped with the embeddings at the lowest rate; anything above the encoder
with the head. Section 13 records why this matters.

Weights are loaded in fp32 for every arm regardless of how the checkpoint is
published, and mixed precision is left to autocast. `mdeberta-v3-base` is
stored in fp16 and declares no dtype in its configuration, so it would
otherwise train in pure half precision with no gradient scaler; besides
diverging, that would have made it the one arm running at a different
numerical precision from the rest. A run whose loss or validation
predictions become non-finite is aborted rather than logged: a diverged run
is not a result, and the training loop previously carried NaN weights to the
end without ever selecting an epoch. Training is on question-level rows with
predictions averaged back per participant, which both multiplies the
training rows and keeps evaluation on the same held-out people as every
other arm.

This replaces the exploratory phase's scheme of cross-validating to find an
epoch budget and refitting on all of the training data. A single grouped
holdout costs one training run instead of six, which is what makes the
560M-parameter arm affordable. The cost is that the validation split is
drawn once rather than varied, that 20 percent of the training participants
are held back from fitting, and
that the stopping signal comes from roughly 22 people — a limitation
Section 13 records, since several runs stopped within the first three
epochs.

## 9c. Cross-Validated Fine-Tuning (E6)

E6 is E5 with one difference: how the epoch budget is chosen, and therefore
how much of the training set reaches the model that is evaluated. Every
other setting -- checkpoint, learning rates, decay, batch size, token limit,
granularity, augmentation -- is held identical, so the contrast is
attributable to the scheme rather than to the recipe.

Procedure: a grouped, stratified 5-fold split of the training participants,
the same splitter notebooks 01-03 use. Each fold trains with early stopping
on its own validation part; the five chosen budgets are pooled with a
median; the evaluated model is refitted on all training rows for that many
epochs with no early stopping. The median rather than the mean because a
fold that stops at epoch 1 -- which the five-seed E5 runs showed happens --
would otherwise drag the budget down for every other fold.

Two quantities are recorded and must not be conflated. The cross-validated
RMSE comes from out-of-fold predictions, aggregated per participant, so no
training row is scored by a model that saw it; it is computed entirely
within the training set and adds no test exposure. The test RMSE comes from
the refitted model on the 48 held-out participants and is what compares with
every other arm.

E6 exists because Section 9b's single holdout has two costs that Section 13
records: 20 percent of training participants never contribute a gradient,
and the stopping signal rests on roughly 22 people. E6 removes the first
entirely and bases the second on the whole training set. It costs k+1
trainings per arm and target instead of one, which is what made it
impractical when e5-large was the binding constraint; the seven-arm grid
makes the comparison worth paying for once.

Whether E6 beats E5 on the held-out participants is left open rather than
predicted. More training data reaching the final model should help; against
that, the refit trains for a fixed budget chosen on other folds and has no
stopping signal of its own, so it can overshoot.

## 10. Statistical Analysis Plan

Seeds: one seed, 42, for every arm. This is a deviation from the five-seed
scheme this protocol was written with, and it is recorded as such rather
than presented as the design.

What the seed controls differs by family. For frozen encoders (E0-E3) the
embedding computation is deterministic and cached, so the seed reaches only
the cross-validation fold assignment and the regressor's own randomness; the
five-seed runs measured seed-to-seed SDs there as low as 0.003 RMSE, and
several arms were exactly reproducible. For fine-tuned encoders (E5) the
seed also sets the head's weight initialization and the validation slice
used for early stopping, and there the same runs measured SDs up to 0.41
RMSE.

The consequence is asymmetric and must be stated wherever these results are
reported: a difference between two frozen arms is close to deterministic and
can be read at face value, while a difference between two fine-tuned arms
smaller than roughly 0.4 RMSE cannot be distinguished from the seed alone.
Reported SD columns are empty by construction. Restoring the estimate means
listing more seeds in `config.SEEDS`; nothing else in the pipeline changes.

Pairing: because every arm is evaluated on the same 48 test participants,
comparisons between two encoders use paired statistics on per-participant
error, not independent-sample tests.

Primary test: for each pairwise encoder comparison, the per-participant
squared-error difference is summarized with a paired bootstrap confidence
interval (2000 resamples) and tested with a Wilcoxon signed-rank test.
Per-participant absolute-error differences are tested the same way as a
robustness check (Section 6).

Multiple-comparison correction: Holm-Bonferroni, applied within each
pre-specified family of comparisons (all encoders against the current best
established baseline for the primary endpoint; all encoders against each
other for secondary endpoints), implemented in `src/metrics.py`,
function `holm_bonferroni`.

Effect size: the mean paired difference and its bootstrap confidence
interval, in questionnaire points, are the effect-size report.

An earlier version of this section also promised Cohen's d on the same
differences. It was never implemented, and it is withdrawn here rather than
added: a standardized difference divides by a spread this design cannot
estimate, since every arm is a single run at one seed (Section 10), so the
number would have carried a precision the study does not have. The interval
in points says the same thing without that implication.

## 11. Secondary Analysis: Regression-Derived vs. Directly Trained Classification

Scope: run once, on the single encoder that wins the primary comparison
(Section 6), not on every arm.

Question: does a classifier trained directly on severity bands with
inverse-frequency class weighting (as already implemented in the
exploratory fine-tuning notebook) detect the minority severe band better
than thresholding the winning regression model's output.

This analysis is reported as an exploratory finding and does not alter the
primary encoder ranking established under Section 6.

## 12. Reproducibility Infrastructure

Environment: this experiment reuses the existing virtual environment at
`../env`, not a separate one. `requirements.txt` in this directory is a
snapshot of that environment for documentation.

Code organization: shared logic (data loading, splitting, metrics, logging,
and the training loop itself) lives in `src/`, imported by both the
notebooks and the batch scripts, so the split, leakage guard, metric
definitions and search procedure exist in exactly one place.

Configuration: settings are split in two, in `src/config.py`. The hold-out
split, its seed, the target definitions and the clinical cutoffs are module
constants — changing any of them breaks comparability with every result
already recorded, so they are not knobs. Everything tunable (seeds, CV
folds, search budget, encoder checkpoints, chunking, TF-IDF preprocessing)
lives in a `TrainingConfig` created at the entry point and passed down
explicitly. Each result row records the `TrainingConfig` it was produced
under (`config_json`) and a timestamp, so results obtained under different
settings can never be silently pooled.

Experiment log: every run appends one row to `experiment_log.csv`
(`src/logging_utils.py`), recording encoder, target, seed and metric values,
plus one row per candidate regressor tried to `model_selection_log.csv` (not
only the one selected), so the model-selection step stays auditable.

Results are laid out as `results/<family>/<augmentation>/<granularity>/`,
each level because the combination changes what is on disk: augmented runs
train on different rows than plain ones, question-level runs embed answers
rather than participants, and fine-tuning produces no embedding cache at
all. What is meant to be compared side by side lives in columns instead —
`search` (defaults or a random search of a given size), `family`,
`granularity` and `notebook` — and `logging_utils.load_all_runs()` returns
the whole tree as one frame, so cross-family comparison is a groupby rather
than a manual concatenation at each call site.

Provenance per row: a run records the checkpoint that produced its features
(`encoder_model`) and the selected model's hyperparameters
(`selected_params`), not only the configuration object. The configuration
alone is insufficient once a single run embeds with several checkpoints:
every e5 arm would otherwise claim the same one.

Embedding caches store the participant id order and the model that produced
them, and refuse to load if either has changed. Both guards exist because a
stale cache is silent: without them, swapping a checkpoint would quietly
reuse the previous model's vectors under the new arm's name.

Exports: `src/reporting.py` writes a dated workbook to `results/exports/`
with sheets for the summary, the clinical read-outs, the best run per seed
and the raw rows. The logs stay CSV because they are appended one row at a
time and a text file does that safely, while rewriting a workbook per row
would be slow and would lose everything if a run died mid-write. The
workbook is for reading and quoting; the CSVs are the record.

Writes to both logs are append-only, preserving the full history of what was
run. Reads go through `logging_utils.load_runs`/`load_trials`, which keep
only the most recent row per (encoder, augmentation, target, seed, search):
re-running a stage supersedes its previous rows rather than being averaged in
with them, while a defaults run and a tuned run of the same arm coexist.

Model artifacts: checkpoints produced under this protocol (the TSDAE
checkpoints, one per saved epoch) are saved under `models/`, unlike the
exploratory phase, where final models were not persisted.

Execution: the notebooks are the primary interface and train one arm per
cell, so a failure costs that arm rather than the whole run. `scripts/`
holds equivalent batch entry points over the same `src/` code, for running a
full sweep unattended.

### Per-participant predictions

Each run also writes its individual test predictions to
`results/<family>/<augmentation>/<granularity>/predictions/`, one file per
`(encoder, target, seed, search)`. A run's RMSE cannot say whether two arms
differ because one is better on the same people or because they fail on
different ones, which is exactly what the paired tests of Section 10
require; those errors are unrecoverable once a run is over, so they are
persisted at the time the run is scored rather than reconstructed later.

## 13. Limitations

- Sample size (n=48 test participants) limits statistical power; small
  true differences between encoders may not reach significance even if
  real.
- Every arm is a single run at seed 42 (Section 10), so no arm carries a
  variance estimate and no reported difference can be checked against the
  measurement noise of the procedure that produced it. The five-seed runs
  this replaces put that noise at up to 0.41 RMSE for the fine-tuned arms,
  which exceeds most of the differences between them; rankings among those
  arms are therefore descriptive only. The frozen arms are far less
  affected, their measured spread having been as low as 0.003.
- The domain-adaptation corpus is small relative to typical TSDAE training
  corpora; E3 results should be interpreted with this constraint in
  mind, and the intrinsic validation step (Section 9) is intended to help
  attribute weak results to either the adaptation step or the downstream
  head.
- Labels are self-reported screening scores, not clinical diagnoses.
- The exploratory phase (Section 2) involved repeated inspection of the
  same test set across several notebooks; this protocol cannot retroactively
  remove that exposure, only avoid extending it further.
- The encoder-by-granularity grid (Section 7) adds ten more evaluations on
  those same 48 participants, and the adaptation curve (Section 9) four
  more. Running each complete and reporting it complete prevents
  cherry-picking, but it does not undo the added exposure, and the best cell
  of a large grid is optimistic by construction.
- The fine-tuned arms early-stop on roughly 22 validation participants
  (Section 9b). Several runs stopped within the first three epochs and
  scored poorly, so part of their spread reflects an unreliable stopping
  signal rather than the method itself. Their seed-to-seed SD (up to 0.41)
  is accordingly much larger than the frozen arms' (as low as 0.003), and
  differences of that size cannot be resolved against it.
- The seven cross-validated arms of Section 9c are seven further
  evaluations on the same 48 test participants, bringing the total to 28.
  They are declared in advance, all reported, and each is paired with its
  E5 counterpart rather than being added to a pool of candidates to pick a
  winner from; that is what keeps the addition a comparison rather than a
  wider search. It remains additional exposure.
- The seven fine-tuned arms (Section 9b) raise the number of arms scored on
  the same 48 test participants from 16 to 21. All seven were declared
  before running and all are reported, which prevents cherry-picking but
  does not reduce the exposure. With between-arm gaps smaller than the
  seed-to-seed spread, the identity of the top-ranked arm is partly a draw
  from that noise; the declared contrasts of Section 9b are read as
  group-level differences, and no conclusion rests on which single
  checkpoint ranks first.
- Two defects in the fine-tuning loop were found while adding the arms of
  Section 9b, both of which would have been read as the affected
  architecture performing badly rather than as implementation faults.
  `mdeberta-v3-base` loaded in the fp16 precision of its published weights
  and went non-finite at the second optimizer step, and the loop tolerated
  that silently: with a non-finite validation score no epoch ever improved
  on the initial best, so early stopping returned the diverged model and the
  failure surfaced only later inside a metric. Both are fixed, and no arm
  was logged under either condition.
- The layer-wise learning-rate decay used before 2026-09-09 assigned rates
  by name and matched only the head, the numbered encoder layers and the
  embeddings. Any parameter outside those three groups was left out of the
  optimizer entirely and therefore frozen. This affected the BERT-family
  pooler, so the first `E5_E5SMALL_FT` results were obtained with a frozen
  pooler and are superseded by a re-run; `E5_E5LARGE_FT`, whose architecture
  has no pooler, is unaffected. Had the defect survived into the arms added
  here it would have frozen mDeBERTa's relative-position embeddings, which
  is the mechanism its architecture is built on, and the resulting failure
  would have been indistinguishable from the architecture performing badly.
  The optimizer now asserts that every parameter is assigned to a group.

## 14. Execution Plan

Phase A: infrastructure. `src/` modules, `requirements.txt`, this protocol.
(Complete.)

All phases below are being re-run from an empty results tree at seed 42; the
five-seed results they replace are archived under `archive/` and are not
read by any analysis path.

Phase B: the frozen encoder grid, participant and question level.

Phase C: TSDAE adaptation, intrinsic validation, and the four checkpoints
evaluated frozen. The adaptation itself is unchanged and its checkpoints are
reused; only the downstream evaluation is re-run. The negative result
recorded in Section 9 was established across five seeds and stands on that
evidence.
End-to-end fine-tuning of the seven checkpoints of Section 9b.

Phase C2: the same seven checkpoints under the cross-validated scheme of
Section 9c, reported paired against their Section 9b counterparts.

Phase D: confirmatory statistical comparison (Section 10), secondary
analysis (Section 11), and reporting.

Phase E, secondary: leave-one-subject-out evaluation of the frozen
approaches, with the downstream model fixed rather than searched per fold,
since nested LOOCV over 158 participants is not affordable. It runs last
and deliberately: LOSO trains and evaluates on all 158 participants, so once
it runs the held-out set is no longer held out for anything after it. Its
numbers are not comparable with the fixed-split results — 157 training
participants against 110, and 158 evaluation points against 48 — and it is
reported as robustness, not as the primary analysis.


## 15. Amendment: named-feature arms (E10-E15) and E5_XLMRLARGE_FT

Added after Sections 1-14 were fixed and after parts of the encoder grid had
already been evaluated. Recorded here as an amendment, with what prompted
each addition, rather than folded into the sections above as though it had
been planned from the start.

### 15.1 What was added

Six arms whose features are named rather than learned, all in the frozen
scheme (features computed once, a classical regressor fitted on top) but
logged under their own results family, `lexical`:

| Arm | Features | Per participant |
|-----|----------|-----------------|
| E10 | lexical categories | 60 |
| E11 | Spanish affective norms | 25 |
| E12 | E10 + E11 | 85 |
| E13 | E10 + TF-IDF (1-3 grams) | 60 + vocabulary |
| E14 | E11 + TF-IDF | 25 + vocabulary |
| E15 | E12 + TF-IDF | 85 + vocabulary |

E0 (TF-IDF alone) completes the factorial but is not re-run: it is measured
in notebook 01 under the full candidate set, which these arms do not use.

The lexical features are person and pro-drop marking read from spaCy
morphology, negation and absolutist terms, PHQ-9/GAD-7 symptom subscales,
morphosyntax, length-robust lexical richness, and spoken-discourse markers
(`src/lexical_features.py`). The affective features are valence, arousal and
five discrete emotions from Stadthagen-Gonzalez et al. (2017, 2018), norms
collected from native Spanish speakers on Spanish words
(`src/affective_norms.py`).

One arm was also added to the fine-tuning family: **E5_XLMRLARGE_FT**
(`xlm-roberta-large`). `intfloat/multilingual-e5-large` *is*
`xlm-roberta-large` with a contrastive pretraining on top, and its only
control in Section 9b was `xlm-roberta-base`, so the gap between them mixed
303M of encoder against 86M with that pretraining against none. The pairing
at equal size separates the two.

### 15.2 Deviations from the procedure of Sections 7 and 10

These arms are **not** comparable row for row with E0-E9. Three things
differ, all recorded in every row's `config_json`:

- **Three candidate regressors instead of nine** (SVR, RandomForest,
  BayesianRidge). Cross-validation still selects among them, but from a
  smaller pool, which is a different selection procedure.
- **Two fold counts, 5 and 10.** The count is part of the selection
  procedure and is written into each row's `search` field, so the two never
  collide on one key. The pair is reported side by side because the gap
  between them turned out to reach 0.50 RMSE on identical features and an
  identical hold-out — twice the margin any of these arms holds over the
  mean predictor, which is itself a result about how much a single
  cross-validated model choice can be trusted at n=110.
- **RidgeCV replaces Ridge** as the linear candidate throughout, for every
  family and not only these arms. With `n_search_iter=0` nothing tunes a
  penalty, and Ridge's default alpha=1.0 reached a cross-validated RMSE of
  25.7 on the 60-feature matrix against 5.94 for the mean predictor. RidgeCV
  chooses alpha over a fixed grid by leave-one-out, solved in closed form, so
  it introduces no nested search. No logged row had ever selected Ridge, so
  no existing result changes meaning. Candidates that cannot read the sparse
  matrix the vectorizer emits were excluded on that ground alone, which ruled
  out BayesianRidge, ARDRegression and PLSRegression.

The mean predictor is now fitted and logged for every arm
(`src/regression_models.py`), so each trial table carries the number its
models had to beat. It is never selected.

### 15.3 Choices made on the training participants

Declared because they were made after seeing data, on the training split
only, and are therefore selection rather than pre-specification:

- **Which 25 affective features.** Chosen after reading their distributions
  and rank correlations with PHQ-9 and GAD-7 over the 110 training
  participants (`scripts/lexical_diagnostic.py --feature-set affect`). Their
  reported correlations should not be read as though the set had been fixed
  in advance.
- **The TF-IDF vocabulary cap (300).** A sweep over 300 / 1000 / 3000 on the
  training participants moved the best cross-validated RMSE by at most 0.07
  points; 300 was kept. `min_df=3` caps the vocabulary near 2,300 n-grams
  whatever the setting, so the 1-3 window yields roughly 218 unigrams, 70
  bigrams and 12 trigrams — these are effectively unigram-and-bigram models.
- **Not using univariate feature selection.** Tested with the selector
  inside the cross-validation fold; keeping every column won in seven of
  eight cells, so no selection step was added.

No decision in this section was taken by looking at the held-out set.

### 15.4 Test-set exposure

Section 13 records 28 evaluations on the same 48 held-out participants. That
count predates the adaptation grids of Sections 9 and 9d and everything in
this amendment, and is no longer accurate; the true figure is several times
larger. The arms here add 24 more (6 arms x 2 granularities x 2 fold counts),
all declared before running and all reported. This does not undo the
exposure, and the caution of Section 13 applies with more force: the best
cell of a grid this size is optimistic by construction, and none of the
differences within it survives Holm correction.
