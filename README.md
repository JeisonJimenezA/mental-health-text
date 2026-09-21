# Text channel — PHQ-9 / GAD-7 from interview transcripts

Confirmatory experiment comparing text representations for predicting PHQ-9
(depression) and GAD-7 (anxiety) severity from transcribed answers to a
semi-structured interview with university students.

`PROTOCOL.md` holds the experimental design, hypotheses and statistical
analysis plan, and was fixed before the experiments it governs. Read it
first: every amendment since records what forced it.

This is the **text channel** of a multimodal study. The audio and video
channels, and their fusion, are separate repositories and out of scope here.

---

## Before anything: what this repository does not contain

**The data is not here and will not be.** The code expects a `data/`
directory *beside* this one, one folder per session (`UTB_*`), holding the
transcriptions and an Excel file with the PHQ-9 and GAD-7 totals:

```
<parent>/
├── data/                  not in this repository
│   └── UTB_.../features/transcription/audio_Q01_transcription.txt
└── text_experiment/       this repository
```

`src/config.py` derives that path and checks it at the point of use, so the
package imports and the tests run without it. Anything that reads a
participant's text will raise with a message saying what is missing.

**`results/*/*/predictions/` does contain individual clinical data.** Each
row is a participant identifier, their real questionnaire score and a model's
prediction. The identifier embeds the session date and time, so it is not
anonymous. These files are needed for the paired tests of `PROTOCOL.md`
Section 10. Treat this repository accordingly, and think before making it
public or adding collaborators.

**Adapted checkpoints are not committed.** `models/` is ~15 GB across 24
files, several above GitHub's 100 MB limit, and they are continued
pretraining of public checkpoints on this study's own clinical transcripts.
`scripts/run_tsdae.py` and `scripts/run_mlm_adaptation.py` rebuild them.

**The affective norms are not redistributable.** Stadthagen-González et al.
(2017, 2018) are supplementary material of their papers. Download them and
place them in `src/resources/`; `src/affective_norms.py` names the two files
and validates their format on load. Without them, only the arms that use
affective features are unavailable.

---

## Layout

```
PROTOCOL.md          experimental design, fixed before running
requirements.txt     direct dependencies, pinned
conftest.py          makes src/ importable from notebooks and tests

src/
  config.py              protocol-frozen constants + the tunable TrainingConfig
  data.py                loading of transcriptions and PHQ-9/GAD-7 scores
  splits.py              fixed train/test split, leakage guard, CV splitter
  encoders.py            TF-IDF (E0), BETO (E1), sentence-transformer (E2/E3)
  lexical_features.py    named lexical features, as composable blocks
  lexical_resources.py   closed word lists and Spanish syllabification
  lexicon.py             PHQ-9/GAD-7 clinical lexicon, by symptom subscale
  affective_norms.py     Spanish valence, arousal and discrete-emotion norms
  regression_models.py   classical regressors and their search space
  metrics.py             regression, clinical and paired-comparison metrics
  logging_utils.py       append-only run and model-selection logs
  pipeline.py            training loop, feature caches, arm registry
  finetuning.py          end-to-end fine-tuning with a regression head
  multitask.py           one encoder, two regression heads
  domain_adaptation.py   TSDAE adaptation
  mlm_adaptation.py      task-adaptive pretraining (MLM)
  reporting.py           spreadsheet exports
  plotting.py            confusion-matrix rendering

notebooks/           one per experiment family; the primary way to run things
scripts/             batch entry points and diagnostics
tests/               unit tests for src/
results/             logs, per-participant predictions, feature caches
```

### Results layout

```
results/<family>/<augmentation>/<granularity>/
```

Each level exists because it changes what is on disk. Augmented runs train on
different rows; question-level runs represent answers rather than
participants; fine-tuning produces no feature cache. Mixing any of them is how
a stale cache or a superseded row goes unnoticed.

Families: `frozen`, `fine-tuning`, `fine-tuning-cv`, `fine-tuning-multitask`,
and `lexical` for the named-feature arms. Each directory holds
`experiment_log.csv` (one row per arm, augmentation, target, seed and search
regime), `model_selection_log.csv` (one row per candidate regressor tried),
`predictions/` and `embeddings/`.

Both logs are **append-only**. `logging_utils.load_runs` keeps the most recent
row per key on read, so re-running a cell adds a row rather than overwriting
one and the history stays auditable.

---

## The arms

| | Representation | Family |
|---|---|---|
| E0 | TF-IDF | frozen |
| E1 | BETO, mean-pooled | frozen |
| E2 | `multilingual-e5`, three sizes | frozen |
| E3 | e5-small adapted with TSDAE | frozen |
| E5 | seven checkpoints fine-tuned end to end | fine-tuning |
| E6 | the same seven, cross-validated epoch budget | fine-tuning-cv |
| E7 | `ELiRF/RoBERTa-es-mental-large` | frozen |
| E8 | MLM-adapted checkpoints, fine-tuned | fine-tuning |
| E9 | one encoder, two regression heads | fine-tuning-multitask |
| **E10-E15** | **named features: lexical, affective, TF-IDF, and their combinations** | **lexical** |

E10-E15 are the subject of `PROTOCOL.md` Section 15, which also records how
they deviate from the procedure of Sections 7 and 10 — a narrower set of
candidate regressors, two fold counts, and decisions taken on the training
participants.

---

## Where each family of representations lands

The lowest PHQ-9 RMSE reached by each kind of representation, on the 48
held-out participants, with that same arm's GAD-7 figure beside it. PHQ-9 is
the primary endpoint (`PROTOCOL.md` Section 6); picking a different winner per
target would be choosing twice from the same test set.

| Representation | Best arm | PHQ-9 RMSE | R² | GAD-7 RMSE | R² |
|---|---|---|---|---|---|
| Frozen, off the shelf | `E2_E5LARGE` (question) | 4.634 | 0.289 | 4.541 | 0.186 |
| Domain-adapted, frozen | `E1_BETO_MLM_EP010` | 4.838 | 0.225 | 4.395 | 0.237 |
| Fine-tuned end to end | `E5_E5LARGE_FT` (question) | **4.263** | **0.398** | 4.860 | 0.067 |
| Named features | `E12_LEXICAL_AFFECT` | 5.166 | 0.116 | 4.995 | 0.014 |

RMSE is in questionnaire points, on scales of 0-27 and 0-21.

The two targets are not on the same footing, which is why a GAD-7 R² can look
low next to a respectable RMSE. R² is measured against each target's own
spread on these 48 participants: 5.495 points for PHQ-9 and 5.031 for GAD-7.
An arm has to beat those to reach R² = 0, so a GAD-7 RMSE of 4.86 is only
just inside the line.

Four things this table does not say, all of which matter more than its
ordering:

- **Each arm is a single run at one seed**, so none of these gaps can be read
  against the noise of the procedure that produced it. Five-seed runs measured
  that noise at up to 0.41 RMSE for the fine-tuned arms (`PROTOCOL.md`
  Sections 10 and 13) — larger than most of the differences here.
- **The rows were not produced by the same procedure.** The named-feature arms
  choose among three candidate regressors, the others among nine, and the
  named-feature arms also report two fold counts. A narrower pool is a
  different selection procedure, not a fairer or worse one (Section 15.2).
- **The winners are drawn from grids**, and the best cell of a grid is
  optimistic by construction. The number of arms scored on these same 48
  participants is large and growing (Section 13).
- **No arm generalizes across both targets.** The best PHQ-9 arm is close to
  the mean predictor on GAD-7, and the best GAD-7 arm is a different one
  entirely. Any claim about "the best representation" has to say for which
  questionnaire.

Full results, including every arm and both fold counts, are under `results/`.

---

## Running

Point a kernel at the shared interpreter:

```
..\env\Scripts\python.exe -m ipykernel install --user --name text-experiment
```

Tests:

```
..\env\Scripts\python.exe tests\test_splits.py
..\env\Scripts\python.exe tests\test_lexical_features.py
..\env\Scripts\python.exe tests\test_affective_features.py
```

Diagnostics, which read the training participants only:

```
..\env\Scripts\python.exe scripts\lexical_diagnostic.py --feature-set lexical
..\env\Scripts\python.exe scripts\vocabulary_diagnostic.py
```

An unattended sweep:

```
..\env\Scripts\python.exe scripts\run_phase_b.py
```

The notebooks are the primary way to run an experiment: they train and report
one arm per cell using the same `src/pipeline.py` code the scripts use, so the
two cannot drift. Every notebook starts by putting this directory on the path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
```

Every run is configured by a `TrainingConfig` created in the notebook's
configuration cell, and it is logged with each result row, so a row always
records the settings that produced it. The hold-out split, its seed and the
clinical cutoffs are deliberately not part of it: the protocol fixes them.

---

## Reading a result

Two cautions apply to every number in `results/`, both from `PROTOCOL.md`
Section 13:

**Every arm is a single run at seed 42**, so no arm carries a variance
estimate and no difference can be read against the noise of the procedure
that produced it. For the fine-tuned arms that noise was measured at up to
0.41 RMSE; for the frozen arms, as low as 0.003.

**The same 48 held-out participants have been used many times.** Section 13
gives a count that predates several grids and is no longer accurate. Every
arm was declared before running and all are reported, which prevents
cherry-picking but does not undo the exposure: the best cell of a grid this
size is optimistic by construction.
