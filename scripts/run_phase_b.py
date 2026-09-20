"""Phase B: E0 (TF-IDF), E1 (BETO) and E2 (e5-small), 5 seeds each, to obtain
a variance estimate for the exploratory baselines (PROTOCOL.md, Sections 7,
10, 14). All the logic lives in src/pipeline.py, shared with
notebooks/01_baseline_encoders.ipynb; this script is a convenience for
running the full sweep unattended (e.g. overnight, or the augmented setting)
without keeping a notebook open.

Usage:
  ..\\env\\Scripts\\python.exe scripts\\run_phase_b.py
  ..\\env\\Scripts\\python.exe scripts\\run_phase_b.py --augmented
  ..\\env\\Scripts\\python.exe scripts\\run_phase_b.py --seeds 42 43 --n-search-iter 20
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src import config, data, encoders, splits
from src.pipeline import arm_models, build_train_frame, compute_frozen_embeddings, \
    compute_lexical_features, encoder_data, encoders_for, \
    lexical_results_dir, run_encoder


def main(cfg: config.TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, config: {cfg}", flush=True)

    df = data.load_dataset()
    train_ids, test_ids = splits.train_test_participants(df)
    train_frame, groups = build_train_frame(df, train_ids, test_ids, cfg)
    test_frame = df.loc[test_ids]
    print(f"train rows: {len(train_frame)}, test participants: {len(test_frame)}", flush=True)

    run_dir = config.results_dir(cfg.augmentation,
                                 granularity=config.GRANULARITY_PARTICIPANT)
    combined = pd.concat([train_frame[config.QUESTIONS], test_frame[config.QUESTIONS]])

    # The named-feature matrices join the dense ones under the same key, since
    # encoder_data reads them all from one mapping, but they are cached in
    # their own family's directory, which is also where their rows are logged.
    # They need no device: the features are counted, not inferred.
    embeddings = compute_frozen_embeddings(combined, device, cfg, run_dir)
    embeddings.update(compute_lexical_features(combined, lexical_results_dir(cfg)))
    for name, matrix in embeddings.items():
        print(f"{name} features: {matrix.shape}", flush=True)

    docs_all = encoders.participant_documents(combined, cfg)

    t_start = time.time()
    models = arm_models(cfg)
    for encoder_name in encoders_for(cfg):
        X_all, preprocessor = encoder_data(encoder_name, combined, docs_all, embeddings, cfg)
        X_train_enc = X_all.loc[train_frame.index]
        X_test_enc = X_all.loc[test_frame.index]
        run_encoder(encoder_name, X_train_enc, X_test_enc, preprocessor,
                    train_frame, test_frame, groups, cfg,
                    encoder_model=models[encoder_name])

    print(f"Phase B total time: {(time.time()-t_start)/60:.1f} min", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true",
                         help="Train on LLM-paraphrase-augmented data "
                              "(results/augmented/) instead of results/no_augmentation/.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS),
                         help="Training/search seeds (default: 5 seeds, per PROTOCOL.md Section 10).")
    parser.add_argument("--n-search-iter", type=int, default=config.N_SEARCH_ITER,
                         help="RandomizedSearchCV iterations per candidate regressor. "
                              "Changing this deviates from the protocol's search procedure.")
    parser.add_argument("--n-cv-splits", type=int, default=config.N_CV_SPLITS)
    parser.add_argument("--sentence-transformer", default=config.SENTENCE_TRANSFORMER_MODEL,
                         help="e5 checkpoint for the E2 arm. The arm id and its embedding cache "
                              "are named after it, so each checkpoint gets its own log rows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config.TrainingConfig(
        augmentation=config.AUGMENTATION_LLM if args.augmented else config.AUGMENTATION_NONE,
        seeds=tuple(args.seeds),
        n_search_iter=args.n_search_iter,
        n_cv_splits=args.n_cv_splits,
        sentence_transformer_model=args.sentence_transformer,
    ))
