"""Phase C, step C2: intrinsic validation of the TSDAE-adapted encoder
(PROTOCOL.md, Section 9).

Compares the adapted encoder against the checkpoint it started from, on
training participants only, before either is used downstream. The point is
diagnostic: if E3 later underperforms, this says whether the adaptation
produced a degraded space or simply did not help.

Reads three things per encoder:
- cosine spread, to detect the collapse failure mode (every vector nearly
  identical), which the reconstruction loss cannot reveal;
- silhouette of the PHQ-9 severity bands;
- rank correlation between embedding distance and PHQ-9 score gap.

Usage:
  ..\\env\\Scripts\\python.exe scripts\\run_intrinsic_validation.py
  ..\\env\\Scripts\\python.exe scripts\\run_intrinsic_validation.py --adapted models\\tsdae_prueba
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src import config, data, splits
from src.domain_adaptation import intrinsic_report, nearest_neighbours


def parse_args():
    parser = argparse.ArgumentParser()
    # The adaptation directory holds one checkpoint per saved epoch, so the
    # default is the final one; the directory itself is not a model.
    parser.add_argument("--adapted", default=str(config.MODELS_DIR / config.TSDAEConfig.output_name
                                                 / f"ep{config.TSDAEConfig.epochs:02d}"),
                         help="Path to one adapted checkpoint (models/<name>/ep<NN>).")
    parser.add_argument("--base", default=config.SENTENCE_TRANSFORMER_MODEL,
                         help="Checkpoint it was adapted from, for comparison.")
    parser.add_argument("--neighbours", type=int, default=3,
                         help="How many participants to show nearest neighbours for.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapted_path = Path(args.adapted)
    if not adapted_path.exists():
        raise SystemExit(f"Adapted checkpoint not found: {adapted_path}\n"
                          "Run scripts/run_tsdae.py first.")

    df = data.load_dataset()
    train_ids, _ = splits.train_test_participants(df)
    print(f"device: {device}, participants: {len(train_ids)} (training only)\n", flush=True)

    reports = []
    for label, model_name in [("base", args.base), ("adapted", str(adapted_path))]:
        print(f"embedding with {label}: {model_name}", flush=True)
        report = intrinsic_report(df, train_ids, model_name, device)
        report["encoder"] = label
        reports.append(report)

    table = pd.DataFrame(reports).set_index("encoder")[
        ["dim", "cosine_mean", "cosine_std", "cosine_p05", "cosine_p95",
         "band_silhouette", "score_distance_spearman", "score_distance_p"]
    ]
    print("\n=== intrinsic comparison (training participants) ===")
    print(table.round(4).to_string())

    print("\n=== nearest neighbours, adapted encoder ===")
    for example in nearest_neighbours(df, train_ids, str(adapted_path), device,
                                       n_examples=args.neighbours):
        print(f"{example['participant']} (PHQ-9 {example['phq9']:.0f})")
        for neighbour in example["neighbours"]:
            print(f"    {neighbour['participant']} (PHQ-9 {neighbour['phq9']:.0f}) "
                  f"cos={neighbour['cosine']:.3f}")

    out_path = adapted_path / "intrinsic_validation.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
