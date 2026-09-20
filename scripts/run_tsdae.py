"""Phase C, step C1: adapt the sentence-transformer to the project's own
transcriptions with TSDAE (PROTOCOL.md, Section 9).

Writes the adapted checkpoint to models/<output_name>/ together with an
adaptation_metadata.json recording the corpus size and every hyperparameter
used, so E3 can state exactly which encoder it evaluated.

The interview corpus is built from the training participants' original
transcriptions only: never the held-out test set, and never the LLM
paraphrases. See src/domain_adaptation.py for why.

Two declared corpus variants, each written to its own directory:
  interviews                 the interviews alone
  interviews+mentalriskes    plus the MentalRiskES patient messages

Usage:
  ..\\env\\Scripts\\python.exe scripts\\run_tsdae.py
  ..\\env\\Scripts\\python.exe scripts\\run_tsdae.py --corpus interviews+mentalriskes
  ..\\env\\Scripts\\python.exe scripts\\run_tsdae.py --corpus interviews+mentalriskes --dry-run
  ..\\env\\Scripts\\python.exe scripts\\run_tsdae.py --epochs 3 --batch-size 16
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import config, data, splits
from src.domain_adaptation import adapt, build_adaptation_corpus

CORPUS_CHOICES = {"+".join(corpora): corpora for corpora in config.TSDAE_CORPORA}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=list(CORPUS_CHOICES), default=config.CORPUS_INTERVIEWS,
                         help="Which texts to adapt on. The output directory follows from it "
                              "unless --output-name is given.")
    parser.add_argument("--base-model", default=config.SENTENCE_TRANSFORMER_MODEL,
                         help="Checkpoint to adapt. Keep it the same as E2's encoder, so "
                              "E3 vs E2 measures adaptation rather than a change of backbone.")
    parser.add_argument("--epochs", type=int, default=config.TSDAEConfig.epochs)
    parser.add_argument("--checkpoint-every-epochs", type=int,
                         default=config.TSDAEConfig.checkpoint_every_epochs,
                         help="Save a checkpoint this often, so adaptation length can be "
                              "read as a curve instead of one fixed budget.")
    parser.add_argument("--batch-size", type=int, default=config.TSDAEConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.TSDAEConfig.learning_rate)
    parser.add_argument("--deletion-ratio", type=float, default=config.TSDAEConfig.deletion_ratio)
    parser.add_argument("--min-sentence-words", type=int,
                         default=config.TSDAEConfig.min_sentence_words)
    parser.add_argument("--max-sentence-words", type=int,
                         default=config.TSDAEConfig.max_sentence_words,
                         help="Run-ons longer than this are split into consecutive pieces.")
    parser.add_argument("--output-name", default=None,
                         help="Directory name under models/. Defaults to the corpus variant's "
                              "own directory (config.TSDAE_CORPORA).")
    parser.add_argument("--seed", type=int, default=config.TSDAEConfig.seed)
    parser.add_argument("--dry-run", action="store_true",
                         help="Build and describe the corpus, then stop without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    corpora = CORPUS_CHOICES[args.corpus]
    cfg = config.TSDAEConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        deletion_ratio=args.deletion_ratio,
        min_sentence_words=args.min_sentence_words,
        max_sentence_words=args.max_sentence_words,
        corpora=corpora,
        output_name=args.output_name or config.TSDAE_CORPORA[corpora],
        seed=args.seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = data.load_dataset()
    train_ids, test_ids = splits.train_test_participants(df)

    if args.dry_run:
        corpus = build_adaptation_corpus(df, train_ids, cfg)
        sentences = [s for source in corpus.values() for s in source]
        lengths = [len(s.split()) for s in sentences]
        print(f"config: {cfg}")
        print(f"output: {config.MODELS_DIR / cfg.output_name}")
        print(f"participants: {len(train_ids)} train (test excluded: {len(test_ids)})")
        for name, source in corpus.items():
            print(f"sentences from {name}: {len(source)}")
        print(f"sentences total: {len(sentences)}")
        if lengths:
            print(f"words per sentence: min {min(lengths)}, median "
                  f"{sorted(lengths)[len(lengths)//2]}, max {max(lengths)}")
            steps = (len(sentences) // cfg.batch_size) * cfg.epochs
            print(f"training steps at batch {cfg.batch_size} x {cfg.epochs} epochs: ~{steps}")
            for name, source in corpus.items():
                print(f"\nfirst 3 sentences from {name}:")
                for s in source[:3]:
                    print(f"  - {s}")
        return

    checkpoints = adapt(df, train_ids, test_ids, cfg, device)
    print("\ncheckpoints saved:")
    for epoch, path in checkpoints.items():
        print(f"  epoch {epoch:>2}  ->  {path}")
    print("\nEvaluate them with notebooks/03_domain_adapted.ipynb.")


if __name__ == "__main__":
    main()
