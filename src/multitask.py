"""Multi-task fine-tuning: one encoder, two regression heads (E9).

Every other fine-tuned arm trains one model per scale, so PHQ-9 and GAD-7 each
see the encoder learn from scratch. Here a single model predicts both at once
and the two errors are backpropagated together.

The reason to expect anything from it is in the labels: PHQ-9 and GAD-7
correlate at r = 0.82 in these participants, and a model's predictions for the
two correlate at 0.97, so the encoder is already learning one shared notion of
distress twice over. Training it once against both doubles the supervised
signal reaching the encoder from the same 110 people, which is the scarce
resource here, and acts as a regularizer on the arm most exposed to
overfitting (PROTOCOL.md, Section 13). It also costs half: one training run
per fold instead of one per scale.

What it is not: using one questionnaire to predict the other. Both labels are
used only while training; at prediction time the model sees text alone, like
every other arm.

Design decisions:

- Equal weight for the two tasks. Targets are standardized separately on the
  fitting slice, so the shared loss is the mean of two comparable errors
  instead of being dominated by PHQ-9's wider range.
- Early stopping on the mean of the two validation RMSEs, so a single budget
  serves both heads; per-scale stopping would be two models again.
- Cross-validated epoch budget and refit on all training rows, exactly as
  src/finetuning.run_finetune_cv_arm, so an arm here is comparable with its
  single-task counterpart in the fine-tuning-cv family.
- Two log rows per run, one per scale, both carrying the same run's epoch
  budget. They are read-outs of one model, which is why they live in their
  own family.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from src import config, logging_utils, metrics, splits
from src.finetuning import TextRegressionDataset, _autocast, build_llrd_optimizer
from src.pipeline import aggregate_predictions, checkpoint_label

SEARCH_LABEL_MT = "finetuned_mt_cv"
TARGETS = [config.PRIMARY_TARGET, config.SECONDARY_TARGET]


class MultiTargetCollator:
    """Pads the tokenized inputs and stacks the two-column targets.

    DataCollatorWithPadding alone cannot: it turns the per-example label into
    a tensor by calling torch.tensor on a list of tensors, which works for a
    scalar target and raises for a vector one.
    """

    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        labels = torch.stack([torch.as_tensor(f["labels"], dtype=torch.float32) for f in features])
        batch = self.pad([{k: v for k, v in f.items() if k != "labels"} for f in features])
        batch["labels"] = labels
        return batch


def build_model(checkpoint: str, device: str):
    """The checkpoint with a fresh two-output regression head.

    `problem_type="regression"` with num_labels=2 makes the model's own loss
    the mean squared error over both outputs, which is the equal-weight
    multi-task objective. fp32 for the same reason as the single-task arms
    (src/finetuning.build_model).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(TARGETS), problem_type="regression", dtype=torch.float32)
    return model.to(device)


@torch.no_grad()
def _predict(model, loader, device, cfg) -> np.ndarray:
    """(n, 2) predictions in standardized units."""
    model.eval()
    outputs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with _autocast(device, cfg):
            logits = model(**batch).logits
        outputs.append(logits.float().cpu().numpy())
    return np.vstack(outputs)


def _loader(texts, targets, tokenizer, cfg, batch_size, shuffle=False):
    dataset = TextRegressionDataset(texts, targets, tokenizer, cfg.max_tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       collate_fn=MultiTargetCollator(tokenizer))


def fit(checkpoint, fit_texts, fit_targets, val_texts, val_targets,
        cfg: config.FineTuningConfig, seed: int, device: str,
        epochs: int | None = None, verbose=True):
    """Trains one two-headed model. `fit_targets` is (n, 2), one column per
    scale in the order of TARGETS.

    With a validation slice it early-stops on the mean of the two RMSEs and
    restores the best weights; with `val_texts=None` it trains for exactly
    `epochs`, which is what the refit after cross-validation needs.

    Returns (model, tokenizer, (mean, std) per column, epochs used,
    best mean validation RMSE in standardized units).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_targets = np.asarray(fit_targets, dtype=float)
    mean = fit_targets.mean(axis=0)
    std = np.where(fit_targets.std(axis=0) > 0, fit_targets.std(axis=0), 1.0)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    fit_loader = _loader(fit_texts, (fit_targets - mean) / std, tokenizer, cfg,
                          cfg.batch_size, shuffle=True)
    if val_texts is None:
        val_loader, val_scaled = None, None
        max_epochs = epochs or cfg.max_epochs
    else:
        val_scaled = (np.asarray(val_targets, dtype=float) - mean) / std
        val_loader = _loader(val_texts, val_scaled, tokenizer, cfg, cfg.eval_batch_size)
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
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{checkpoint}: loss became non-finite at epoch {epoch}, step {step}. "
                    f"Training diverged; lower lr_head and lr_encoder_top.")
            loss.backward()
            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if val_loader is None:
            best_epoch = epoch
            if verbose:
                print(f"    epoch {epoch:>2}  (refit, no early stopping)", flush=True)
            continue

        val_pred = _predict(model, val_loader, device, cfg)
        per_target = np.sqrt(np.mean((val_pred - val_scaled) ** 2, axis=0))
        val_rmse = float(per_target.mean())
        if not np.isfinite(val_rmse):
            raise RuntimeError(
                f"{checkpoint}: validation predictions are non-finite after epoch {epoch}.")
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, since_improved = val_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1
        if verbose:
            scores = "  ".join(f"{t} {r:.4f}" for t, r in zip(TARGETS, per_target))
            print(f"    epoch {epoch:>2}  val_rmse(scaled) mean {val_rmse:.4f}  [{scores}]"
                  f"{'  *' if since_improved == 0 else ''}", flush=True)
        if since_improved >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, tokenizer, (mean, std), best_epoch,
            best_rmse if val_loader is not None else float("nan"))


def _predict_texts(model, tokenizer, texts, scaling, cfg, device) -> np.ndarray:
    """(n, 2) predictions in each scale's own units."""
    mean, std = scaling
    loader = _loader(texts, np.zeros((len(texts), len(TARGETS))), tokenizer, cfg,
                      cfg.eval_batch_size)
    return _predict(model, loader, device, cfg) * std + mean


def run_multitask_cv_arm(arm: str, checkpoint: str, q_train, q_test, groups, test_groups,
                          cfg: config.FineTuningCVConfig, device: str,
                          verbose=True) -> list[dict]:
    """One checkpoint, both scales at once, with the cross-validated epoch
    budget and refit of src/finetuning.run_finetune_cv_arm.

    Returns one row per scale, both from the same training run. The folds
    choose a single budget from the shared validation score, so the two
    read-outs cannot drift apart into two different models.
    """
    train_texts = q_train["text"].to_numpy()
    test_texts = q_test["text"].to_numpy()
    y_train = np.column_stack([q_train[config.TARGET_COLUMNS[t]].to_numpy(dtype=float)
                                for t in TARGETS])
    y_test_rows = np.column_stack([q_test[config.TARGET_COLUMNS[t]].to_numpy(dtype=float)
                                    for t in TARGETS])
    strat = splits.stratify_key(q_train)
    run_rows = []

    for seed in cfg.seeds:
        folds = list(splits.cv_splitter(groups=groups, n_splits=cfg.n_cv_splits,
                                         random_state=seed).split(train_texts, strat, groups=groups))
        t0 = time.time()
        if verbose:
            print(f"  {arm} seed={seed} targets={'+'.join(TARGETS)}: "
                  f"{len(folds)} folds + refit", flush=True)

        oof = np.full_like(y_train, np.nan, dtype=float)
        fold_epochs, trial_rows = [], []
        for k, (fit_idx, val_idx) in enumerate(folds, start=1):
            if verbose:
                print(f"   fold {k}/{len(folds)}  fit {len(fit_idx)} rows / "
                      f"{len(set(groups[fit_idx]))} people", flush=True)
            model, tokenizer, scaling, best_epoch, _ = fit(
                checkpoint, train_texts[fit_idx], y_train[fit_idx],
                train_texts[val_idx], y_train[val_idx], cfg, seed, device, verbose=verbose)
            oof[val_idx] = _predict_texts(model, tokenizer, train_texts[val_idx],
                                           scaling, cfg, device)
            fold_epochs.append(best_epoch)
            for column, target in enumerate(TARGETS):
                fold_rmse = float(np.sqrt(np.mean((oof[val_idx, column] - y_train[val_idx, column]) ** 2)))
                trial_rows.append({
                    "encoder": arm, "augmentation": cfg.augmentation, "target": target,
                    "seed": seed, "model_name": f"fold{k}_epoch{best_epoch}",
                    "cv_rmse": fold_rmse, "is_selected": False,
                    "encoder_model": checkpoint_label(checkpoint), "search": SEARCH_LABEL_MT,
                    "family": config.FAMILY_FINETUNING_MT,
                    "granularity": config.GRANULARITY_QUESTION, "notebook": "07",
                })
            del model
            torch.cuda.empty_cache()

        assert not np.isnan(oof).any(), "every training row must be scored out of fold"
        refit_epochs = max(1, int(np.median(fold_epochs)))
        if verbose:
            print(f"   fold epochs {fold_epochs} -> refit on all {len(train_texts)} rows "
                  f"for {refit_epochs}", flush=True)
        model, tokenizer, scaling, _, _ = fit(
            checkpoint, train_texts, y_train, None, None, cfg, seed, device,
            epochs=refit_epochs, verbose=verbose)
        test_pred_rows = _predict_texts(model, tokenizer, test_texts, scaling, cfg, device)

        for column, target in enumerate(TARGETS):
            cv_pred, cv_true, _ = aggregate_predictions(oof[:, column], y_train[:, column], groups)
            cv = metrics.regression_metrics(cv_true, metrics.clip_to_valid_range(cv_pred, target))

            y_pred, y_test, participants = aggregate_predictions(
                test_pred_rows[:, column], y_test_rows[:, column], test_groups)
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
                "selected_model": "finetuned_multitask",
                "selected_params": {
                    "fold_epochs": [int(e) for e in fold_epochs],
                    "refit_epochs": refit_epochs,
                    "cv_rmse": round(cv["rmse"], 4),
                    "cv_mae": round(cv["mae"], 4),
                    "cv_r2": round(cv["r2"], 4),
                    "heads": TARGETS,
                },
                "encoder_model": checkpoint_label(checkpoint),
                "search": SEARCH_LABEL_MT,
                "family": config.FAMILY_FINETUNING_MT,
                "granularity": config.GRANULARITY_QUESTION,
                "notebook": "07",
                "notes": "one model, both scales",
            }
            logging_utils.log_run(row, cfg)
            logging_utils.log_predictions(row, participants, y_test, y_pred)
            run_rows.append(row)
            if verbose:
                print(f"    -> {target}: [CV] rmse {cv['rmse']:.3f}   [TEST] rmse {reg['rmse']:.3f}  "
                      f"mae {reg['mae']:.3f}  r2 {reg['r2']:.3f}", flush=True)

        logging_utils.log_trials(trial_rows, cfg)
        del model
        torch.cuda.empty_cache()
        if verbose:
            print(f"    ({(time.time() - t0) / 60:.1f} min for both scales)", flush=True)
    return run_rows
