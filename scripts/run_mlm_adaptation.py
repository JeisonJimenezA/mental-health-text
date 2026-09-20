"""Task-adaptive pretraining (MLM) of a Spanish encoder on this study's text.

Continues a checkpoint's masked-language-model pretraining on the training
participants' interview answers, optionally with the MentalRiskES patient
messages, then saves the declared checkpoints for downstream evaluation
(frozen against its unadapted arm, fine-tuned against the E5/E8 arms). See
src/mlm_adaptation.py for the data and masking decisions.

Two backbones, two corpora, two masking schemes; each combination writes to
its own directory under models/, so no run can overwrite another:

  --base-model beto              BETO, whose arms are already measured
  --base-model robertaes-mental  ELiRF/RoBERTa-es-mental-large, already
                                 domain-adapted by its authors, so this is
                                 the DAPT-then-TAPT combination
  --corpus interviews | interviews+mentalriskes
  --lexicon-masking              mask clinical-lexicon words more often

Defaults follow the backbone: the 355M RoBERTa trains at batch 2 with 16
accumulation steps (the same effective batch as BETO's 8x4, which is what
fits in 8GB) and overrides its inherited dropout of 0.0, a setting meant for
a 570GB pretraining corpus rather than 400 documents.

Usage:
  ..\\env\\Scripts\\python.exe scripts\\run_mlm_adaptation.py --dry-run
  ..\\env\\Scripts\\python.exe scripts\\run_mlm_adaptation.py
  ..\\env\\Scripts\\python.exe scripts\\run_mlm_adaptation.py --corpus interviews+mentalriskes
  ..\\env\\Scripts\\python.exe scripts\\run_mlm_adaptation.py --base-model robertaes-mental
  ..\\env\\Scripts\\python.exe scripts\\run_mlm_adaptation.py --base-model robertaes-mental \\
      --corpus interviews+mentalriskes --lexicon-masking
"""
import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers.utils import logging as hf_logging

from src import config, data, splits
from src.mlm_adaptation import WholeWordMaskCollator, adapt, prepare

# Length checks tokenize whole answers before they are split, which makes the
# tokenizer warn about sequences over 512 tokens that never reach the model.
hf_logging.set_verbosity_error()

CORPUS_CHOICES = {"+".join(corpora): corpora for corpora in config.TSDAE_CORPORA}
DEFAULTS = config.MLMConfig()

# What a backbone needs beyond the shared recipe. The effective batch stays
# at 32 for both, so the two runs differ in the checkpoint and not in the
# optimization; only how many examples fit on the GPU at once changes.
BACKBONE_DEFAULTS = {
    "beto": {},
    "robertaes-mental": {"batch_size": 2, "grad_accum_steps": 16, "dropout": 0.1},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", choices=list(config.MLM_BACKBONES), default="beto",
                         help="Checkpoint to adapt.")
    parser.add_argument("--corpus", choices=list(CORPUS_CHOICES), default=config.CORPUS_INTERVIEWS)
    parser.add_argument("--lexicon-masking", action="store_true",
                         help="Mask clinical-lexicon words more often (src/lexicon.py), "
                              "keeping the overall masking rate.")
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+",
                         default=list(DEFAULTS.checkpoint_epochs))
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Defaults to the backbone's own setting.")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                         help="Defaults to the backbone's own setting.")
    parser.add_argument("--dropout", type=float, default=None,
                         help="Override the checkpoint's dropout. Defaults to the backbone's "
                              "own setting; omit to keep whatever the checkpoint declares.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--mask-probability", type=float, default=DEFAULTS.mask_probability)
    parser.add_argument("--lexicon-mask-probability", type=float,
                         default=DEFAULTS.lexicon_mask_probability)
    parser.add_argument("--output-name", default=None,
                         help="Directory under models/. Defaults to the variant's own name.")
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--dry-run", action="store_true",
                         help="Prepare and describe the data and masking, then stop without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    corpora = CORPUS_CHOICES[args.corpus]
    base_model = config.MLM_BACKBONES[args.base_model]
    backbone = dict(BACKBONE_DEFAULTS[args.base_model])
    for name in ("batch_size", "grad_accum_steps", "dropout"):
        if getattr(args, name) is not None:
            backbone[name] = getattr(args, name)

    cfg = config.MLMConfig(
        base_model=base_model,
        corpora=corpora,
        lexicon_masking=args.lexicon_masking,
        epochs=args.epochs,
        checkpoint_epochs=tuple(args.checkpoint_epochs),
        learning_rate=args.learning_rate,
        mask_probability=args.mask_probability,
        lexicon_mask_probability=args.lexicon_mask_probability,
        output_name=args.output_name or config.mlm_output_name(
            corpora, args.lexicon_masking, base_model),
        seed=args.seed,
        **backbone,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = data.load_dataset()
    train_ids, test_ids = splits.train_test_participants(df)

    if args.dry_run:
        prepared = prepare(df, train_ids, test_ids, cfg)
        n_train = len(prepared["train_set"])
        steps = math.ceil(math.ceil(n_train / cfg.batch_size) / cfg.grad_accum_steps)
        print(f"config: {cfg}")
        print(f"output: {config.MODELS_DIR / cfg.output_name}")
        print(f"device: {device}")
        print(f"participants: {len(train_ids)} train (test excluded: {len(test_ids)})")
        print("held out for validation: " + "; ".join(
            f"{source}: {len(groups)}" for source, groups in prepared["held_out"].items()))
        print(f"train: {prepared['train_summary']}")
        print(f"validation: {prepared['val_summary']}")
        lengths = prepared["train_set"]["length"]
        print(f"tokens per document: min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, "
              f"max {max(lengths)}")
        print(f"optimizer steps: {steps} per epoch, {steps * cfg.epochs} total "
              f"(effective batch {cfg.batch_size * cfg.grad_accum_steps})")
        print("realized masking: " + ", ".join(f"{k} {v:.3f}" for k, v in prepared["masking"].items()))
        collator = WholeWordMaskCollator(prepared["tokenizer"], cfg, seed=cfg.seed)
        for source in cfg.corpora:
            index = next(i for i, s in enumerate(prepared["train_set"]["source"]) if s == source)
            print(f"\nexample from {source} (<targets>):")
            show_masked(prepared["tokenizer"], prepared["train_set"][index], collator)
        return

    checkpoints = adapt(df, train_ids, test_ids, cfg, device)
    print("\ncheckpoints saved:")
    for epoch, path in checkpoints.items():
        print(f"  epoch {epoch:>3}  ->  {path}")
    print(f"\nLoss history: {config.MODELS_DIR / cfg.output_name / 'mlm_history.json'}")


def show_masked(tokenizer, example, collator, passes=2):
    """The same document masked twice, to show masks change between epochs.

    Each piece is decoded on its own rather than printed as a raw token, so
    byte-level vocabularies (RoBERTa) read as Spanish instead of as their
    byte escapes, and WordPiece continuations keep their ## marker.
    """
    for i in range(passes):
        ids, labels = collator.mask(example)
        pieces = []
        for token_id, label in zip(ids[1:60].tolist(), labels[1:60].tolist()):
            piece = tokenizer.decode([token_id]).strip() or tokenizer.convert_ids_to_tokens(token_id)
            pieces.append(f"<{piece}>" if label != -100 else piece)
        print(f"  pass {i + 1}: {' '.join(pieces)} ...")


if __name__ == "__main__":
    main()
