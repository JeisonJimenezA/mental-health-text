"""End-to-end fine-tuning of an encoder with a regression head (E4/E5).

The frozen arms leave the encoder untouched: it emits vectors and a separate
classical model learns from them, so the transformer never learns anything
about depression severity. Here the prediction error backpropagates through
the whole encoder, which reorganizes its representation for this target.
It is the only approach in the study that optimizes the actual objective
rather than a semantic-similarity proxy, and the one most exposed to
overfitting: a few hundred training rows against 10^8 parameters.

Three mitigations, carried over from the exploratory phase:

- layer-wise learning-rate decay, so the lower and more general layers move
  least and the pretrained knowledge is disturbed as little as possible;
- early stopping on a validation slice split by participant, never by row,
  so no one's answers sit on both sides of the split;
- training on question-level rows with predictions averaged back per
  participant, which both multiplies the training rows and keeps evaluation
  on the same held-out people as every other arm.

The training loop is written out rather than delegated to a framework
trainer: the best weights are kept in CPU memory instead of checkpointed to
disk each epoch, which for a 560M-parameter encoder saves tens of gigabytes
of writes per run.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config, logging_utils, metrics, splits
from src.pipeline import aggregate_predictions, checkpoint_label

# Recorded on every cross-validated row, so a groupby never pools these
# with the single-holdout runs of notebook 04.
SEARCH_LABEL_CV = "finetuned_cv"


class TextRegressionDataset(Dataset):
    """Tokenized answers with their participant's score as the target."""

    def __init__(self, texts, targets, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets[index])
        return item


def build_model(checkpoint: str, device: str):
    """Loads a checkpoint with a fresh single-output regression head.

    The weights are forced to fp32. Transformers infers the dtype from the
    checkpoint file when the configuration does not declare one, and some
    checkpoints are published in half precision: `mdeberta-v3-base` is
    stored in fp16, so it would otherwise train in pure fp16 with no
    gradient scaler and go non-finite within two steps. Master weights stay
    fp32 for every arm and mixed precision is applied by autocast instead,
    which is both numerically sound and the only way these arms differ in
    nothing but the checkpoint.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=1, problem_type="regression", dtype=torch.float32)
    return model.to(device)


def build_llrd_optimizer(model, cfg: config.FineTuningConfig):
    """AdamW with a lower learning rate the deeper into the encoder a
    parameter sits: the head learns fastest, the top encoder layer starts at
    lr_encoder_top, and each layer below it is scaled by llrd_decay.

    Assignment is by position in the network rather than by name, because
    the architectures compared here do not agree on what a base model
    contains. BERT-family models carry a pooler between the encoder and the
    head; DeBERTa keeps relative-position embeddings and a normalization
    layer on the encoder itself, shared by every layer. Anything shared
    across layers is grouped with the embeddings at the lowest rate, since
    it is low-level machinery the pretrained weights depend on; anything
    sitting above the encoder is grouped with the head.

    The final assertion is the point of writing it this way: an earlier
    version collected only the head, the numbered layers and the embeddings,
    which silently left every other parameter out of the optimizer
    altogether. A frozen pooler merely wastes capacity, but DeBERTa's
    relative-position embeddings are what its attention mechanism is built
    on, and freezing those would have looked like the architecture failing.
    """
    base = model.base_model
    prefix = model.base_model_prefix
    layers = list(base.encoder.layer)

    def owned_by(name: str) -> str:
        if not name.startswith(f"{prefix}."):
            return "head"
        inner = name[len(prefix) + 1:]
        if inner.startswith("embeddings."):
            return "embeddings"
        if inner.startswith("encoder.layer."):
            return f"layer{int(inner.split('.')[2])}"
        # Shared across the whole encoder (DeBERTa's rel_embeddings and
        # LayerNorm) versus sitting on top of it (BERT's pooler).
        return "embeddings" if inner.startswith("encoder.") else "head"

    buckets: dict[str, list] = {}
    for name, param in model.named_parameters():
        buckets.setdefault(owned_by(name), []).append(param)

    rates = {"head": cfg.lr_head}
    lr = cfg.lr_encoder_top
    for index in reversed(range(len(layers))):
        rates[f"layer{index}"] = lr
        lr *= cfg.llrd_decay
    rates["embeddings"] = lr

    groups = [{"params": buckets[name], "lr": rates[name]}
               for name in rates if name in buckets]

    assigned = sum(len(group["params"]) for group in groups)
    total = sum(1 for _ in model.parameters())
    assert assigned == total, f"{total - assigned} parameters left unoptimized"

    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _autocast(device: str, cfg: config.FineTuningConfig):
    enabled = cfg.use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def _predict(model, loader, device, cfg):
    model.eval()
    outputs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with _autocast(device, cfg):
            logits = model(**batch).logits
        outputs.append(logits.float().squeeze(-1).cpu().numpy())
    return np.concatenate(outputs)


def train_one(checkpoint, train_texts, train_targets, train_groups,
               cfg: config.FineTuningConfig, seed: int, device: str, verbose=True):
    """Fine-tunes one model, early-stopping on a participant-level validation
    slice. Returns (model, tokenizer, target mean/std, epoch stopped at).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction, random_state=seed)
    fit_idx, val_idx = next(splitter.split(train_texts, train_targets, groups=train_groups))
    return fit(checkpoint, train_texts[fit_idx], train_targets[fit_idx],
                train_texts[val_idx], train_targets[val_idx], cfg, seed, device,
                verbose=verbose)


def fit(checkpoint, fit_texts, fit_targets, val_texts, val_targets,
         cfg: config.FineTuningConfig, seed: int, device: str,
         epochs: int | None = None, verbose=True):
    """Trains one model on an explicit fit slice.

    With a validation slice, training early-stops on it and the best weights
    are restored. With `val_texts=None` it trains for exactly `epochs` and
    keeps the final weights, which is what a refit after cross-validation
    needs: the folds have already decided the budget, and there is no held-
    back data left to stop on.

    Returns (model, tokenizer, target mean/std, epochs used, best val RMSE
    in standardized units, or NaN when there was no validation slice).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Targets are standardized on the fitting slice only, so the validation
    # split cannot influence the scaling.
    mean = float(np.mean(fit_targets))
    std = float(np.std(fit_targets)) or 1.0

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)

    fit_ds = TextRegressionDataset(fit_texts, (fit_targets - mean) / std,
                                    tokenizer, cfg.max_tokens)
    fit_loader = DataLoader(fit_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    if val_texts is None:
        val_ds = val_loader = None
        max_epochs = epochs or cfg.max_epochs
    else:
        val_ds = TextRegressionDataset(val_texts, (val_targets - mean) / std,
                                        tokenizer, cfg.max_tokens)
        val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, collate_fn=collator)
        max_epochs = cfg.max_epochs

    model = build_model(checkpoint, device)
    optimizer = build_llrd_optimizer(model, cfg)
    total_steps = max(1, (len(fit_loader) // cfg.grad_accum_steps) * max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

    best_rmse, best_state, best_epoch, since_improved = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(fit_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device, cfg):
                loss = model(**batch).loss / cfg.grad_accum_steps
            # A diverged run is not a result. Without this the loop would
            # carry NaN weights to the end, never beat the initial best
            # score, return the diverged model, and surface hundreds of
            # lines later as a NaN inside a scikit-learn metric.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{checkpoint}: loss became non-finite at epoch {epoch}, "
                    f"step {step}. Training diverged; lower lr_head and "
                    f"lr_encoder_top, or check the checkpoint's dtype.")
            loss.backward()
            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if val_loader is None:
            # No validation slice: the epoch budget is already fixed, so
            # every epoch simply runs and the final weights are kept.
            best_epoch = epoch
            if verbose:
                print(f"    epoch {epoch:>2}  (refit, no early stopping)", flush=True)
            continue

        val_pred = _predict(model, val_loader, device, cfg)
        val_rmse = float(np.sqrt(np.mean((val_pred - val_ds.targets) ** 2)))
        if not np.isfinite(val_rmse):
            raise RuntimeError(
                f"{checkpoint}: validation predictions are non-finite after "
                f"epoch {epoch}, so no epoch can be selected.")
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, since_improved = val_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1
        if verbose:
            print(f"    epoch {epoch:>2}  val_rmse(scaled) {val_rmse:.4f}"
                  f"{'  *' if since_improved == 0 else ''}", flush=True)
        if since_improved >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    # A refit has no validation score to report; float("inf") would be
    # written to the log as if it were one.
    return (model, tokenizer, (mean, std), best_epoch,
            best_rmse if val_loader is not None else float("nan"))


def run_finetune_arm(arm: str, checkpoint: str, q_train, q_test, groups, test_groups,
                      cfg: config.FineTuningConfig, device: str,
                      targets=None, verbose=True) -> list[dict]:
    """Fine-tunes one checkpoint across seeds and targets, logging each run
    under the fine-tuning results tree. Evaluation is participant level, so
    these rows compare directly with every frozen arm.
    """
    from transformers import DataCollatorWithPadding

    targets = targets or [config.PRIMARY_TARGET, config.SECONDARY_TARGET]
    train_texts = q_train["text"].to_numpy()
    test_texts = q_test["text"].to_numpy()
    run_rows = []

    for seed in cfg.seeds:
        for target in targets:
            column = config.TARGET_COLUMNS[target]
            y_train = q_train[column].to_numpy(dtype=float)
            y_test_rows = q_test[column].to_numpy(dtype=float)

            t0 = time.time()
            if verbose:
                print(f"  {arm} seed={seed} target={target}", flush=True)
            model, tokenizer, (mean, std), best_epoch, best_val = train_one(
                checkpoint, train_texts, y_train, groups, cfg, seed, device, verbose)

            collator = DataCollatorWithPadding(tokenizer)
            test_ds = TextRegressionDataset(test_texts, (y_test_rows - mean) / std,
                                             tokenizer, cfg.max_tokens)
            test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, collate_fn=collator)
            y_pred_rows = _predict(model, test_loader, device, cfg) * std + mean

            # Question-level predictions averaged back per participant, then
            # clipped to the scale's clinical range, exactly as the frozen arms.
            y_pred, y_test, participants = aggregate_predictions(
                y_pred_rows, y_test_rows, test_groups)
            y_pred = metrics.clip_to_valid_range(y_pred, target)

            reg = metrics.regression_metrics(y_test, y_pred)
            clinical = metrics.clinical_utility_metrics(y_test, y_pred)
            row = {
                "encoder": arm, "augmentation": cfg.augmentation, "target": target,
                "seed": seed, "fold": "test", "split": "confirmatory",
                **reg,
                "band_f1_macro": clinical["band_f1_macro"],
                "band_qwk": clinical["band_qwk"],
                "band_confusion_matrix": clinical["band_confusion_matrix"],
                "screening_f1": clinical["screening_f1"],
                "screening_auc": clinical.get("screening_auc", ""),
                "screening_confusion_matrix": clinical["screening_confusion_matrix"],
                "selected_model": "finetuned",
                "selected_params": {"stopped_epoch": best_epoch,
                                     "best_val_rmse_scaled": round(best_val, 4)},
                "encoder_model": checkpoint_label(checkpoint),
                # Not "n/a": pandas reads that back as NaN and the label is lost.
                "search": "finetuned",   # no downstream regressor is selected here
                "family": config.FAMILY_FINETUNING,
                "granularity": config.GRANULARITY_QUESTION,
                "notebook": "04",
                "notes": "",
            }
            logging_utils.log_run(row, cfg)
            logging_utils.log_predictions(row, participants, y_test, y_pred)
            run_rows.append(row)

            del model
            torch.cuda.empty_cache()
            if verbose:
                print(f"    -> rmse {reg['rmse']:.3f}  mae {reg['mae']:.3f}  r2 {reg['r2']:.3f}"
                      f"  (best epoch {best_epoch}, {time.time()-t0:.0f}s)", flush=True)
    return run_rows


def _predict_texts(model, tokenizer, texts, scaling, cfg, device):
    """Predictions in the target's own units for a set of raw texts."""
    from transformers import DataCollatorWithPadding

    mean, std = scaling
    dataset = TextRegressionDataset(texts, np.zeros(len(texts)), tokenizer, cfg.max_tokens)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size,
                         collate_fn=DataCollatorWithPadding(tokenizer))
    return _predict(model, loader, device, cfg) * std + mean


def run_finetune_cv_arm(arm: str, checkpoint: str, q_train, q_test, groups, test_groups,
                         cfg: config.FineTuningCVConfig, device: str,
                         targets=None, verbose=True) -> list[dict]:
    """Fine-tunes one checkpoint with an inner cross-validation over the
    training participants, then refits on all of them.

    The folds are grouped and stratified exactly as in notebooks 01-03, and
    they decide one thing: the number of epochs. Each fold early-stops on
    its own validation part; the fold budgets are pooled with a median, which
    is used because a single fold that stops at epoch 1 should not drag the
    budget down the way a mean would; and the model that is evaluated is
    refitted on every training row for that many epochs.

    Two numbers come out of this and they must not be confused. The
    cross-validated RMSE is computed from out-of-fold predictions, so no
    training row is scored by a model that saw it, and it never touches the
    test set. The test RMSE comes from the refitted model on the 48 held-out
    participants and is the one that compares with every other arm.
    """
    targets = targets or [config.PRIMARY_TARGET, config.SECONDARY_TARGET]
    train_texts = q_train["text"].to_numpy()
    test_texts = q_test["text"].to_numpy()
    strat = splits.stratify_key(q_train)
    run_rows = []

    for seed in cfg.seeds:
        splitter = splits.cv_splitter(groups=groups, n_splits=cfg.n_cv_splits,
                                       random_state=seed)
        folds = list(splitter.split(train_texts, strat, groups=groups))

        for target in targets:
            column = config.TARGET_COLUMNS[target]
            y_train = q_train[column].to_numpy(dtype=float)
            y_test_rows = q_test[column].to_numpy(dtype=float)

            t0 = time.time()
            if verbose:
                print(f"  {arm} seed={seed} target={target}: "
                      f"{len(folds)} folds + refit", flush=True)

            oof = np.full(len(train_texts), np.nan)
            fold_epochs, trial_rows = [], []
            for k, (fit_idx, val_idx) in enumerate(folds, start=1):
                if verbose:
                    print(f"   fold {k}/{len(folds)}  "
                          f"fit {len(fit_idx)} rows / {len(set(groups[fit_idx]))} people", flush=True)
                model, tokenizer, scaling, best_epoch, _ = fit(
                    checkpoint, train_texts[fit_idx], y_train[fit_idx],
                    train_texts[val_idx], y_train[val_idx], cfg, seed, device, verbose=verbose)
                oof[val_idx] = _predict_texts(model, tokenizer, train_texts[val_idx],
                                               scaling, cfg, device)
                fold_epochs.append(best_epoch)
                fold_rmse = float(np.sqrt(np.mean((oof[val_idx] - y_train[val_idx]) ** 2)))
                trial_rows.append({
                    "encoder": arm, "augmentation": cfg.augmentation, "target": target,
                    "seed": seed, "model_name": f"fold{k}_epoch{best_epoch}",
                    "cv_rmse": fold_rmse, "is_selected": False,
                    "encoder_model": checkpoint_label(checkpoint), "search": SEARCH_LABEL_CV,
                    "family": config.FAMILY_FINETUNING_CV,
                    "granularity": config.GRANULARITY_QUESTION, "notebook": "05",
                })
                del model
                torch.cuda.empty_cache()

            assert not np.isnan(oof).any(), "every training row must be scored out of fold"
            # Out-of-fold predictions, aggregated per participant so the CV
            # number is on the same scale as the test number.
            cv_pred, cv_true, _ = aggregate_predictions(oof, y_train, groups)
            cv_pred = metrics.clip_to_valid_range(cv_pred, target)
            cv = metrics.regression_metrics(cv_true, cv_pred)

            refit_epochs = max(1, int(np.median(fold_epochs)))
            if verbose:
                print(f"   fold epochs {fold_epochs} -> refit on all "
                      f"{len(train_texts)} rows for {refit_epochs}", flush=True)
            model, tokenizer, scaling, _, _ = fit(
                checkpoint, train_texts, y_train, None, None, cfg, seed, device,
                epochs=refit_epochs, verbose=verbose)
            y_pred_rows = _predict_texts(model, tokenizer, test_texts, scaling, cfg, device)

            y_pred, y_test, participants = aggregate_predictions(
                y_pred_rows, y_test_rows, test_groups)
            y_pred = metrics.clip_to_valid_range(y_pred, target)

            reg = metrics.regression_metrics(y_test, y_pred)
            clinical = metrics.clinical_utility_metrics(y_test, y_pred)
            row = {
                "encoder": arm, "augmentation": cfg.augmentation, "target": target,
                "seed": seed, "fold": "test", "split": "confirmatory",
                **reg,
                "band_f1_macro": clinical["band_f1_macro"],
                "band_qwk": clinical["band_qwk"],
                "band_confusion_matrix": clinical["band_confusion_matrix"],
                "screening_f1": clinical["screening_f1"],
                "screening_auc": clinical.get("screening_auc", ""),
                "screening_confusion_matrix": clinical["screening_confusion_matrix"],
                "selected_model": "finetuned_cv",
                "selected_params": {
                    "fold_epochs": [int(e) for e in fold_epochs],
                    "refit_epochs": refit_epochs,
                    # Training-data-only estimates, kept beside the test
                    # metrics so the two are never read as one number.
                    "cv_rmse": round(cv["rmse"], 4),
                    "cv_mae": round(cv["mae"], 4),
                    "cv_r2": round(cv["r2"], 4),
                },
                "encoder_model": checkpoint_label(checkpoint),
                "search": SEARCH_LABEL_CV,
                "family": config.FAMILY_FINETUNING_CV,
                "granularity": config.GRANULARITY_QUESTION,
                "notebook": "05",
                "notes": "",
            }
            logging_utils.log_trials(trial_rows, cfg)
            logging_utils.log_run(row, cfg)
            logging_utils.log_predictions(row, participants, y_test, y_pred)
            run_rows.append(row)

            del model
            torch.cuda.empty_cache()
            if verbose:
                print(f"    -> [CV] rmse {cv['rmse']:.3f}   "
                      f"[TEST] rmse {reg['rmse']:.3f}  mae {reg['mae']:.3f}  "
                      f"r2 {reg['r2']:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    return run_rows
