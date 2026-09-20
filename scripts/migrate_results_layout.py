"""One-off migration to the results/<family>/<augmentation>/<granularity>/ layout.

Before, every frozen arm wrote to one log and one embedding directory, with
participant-level and question-level runs told apart only by a `_QUESTION`
suffix in the arm name and a `_questions` suffix in the cache filename. This
moves each row and each cache into the directory that describes it and fills
the new family/granularity/notebook columns.

The rule is deterministic (the arm name says which granularity a row is), so
the migration verifies itself: it compares row counts and every metric value
before and after, and refuses to delete the originals unless they match.

Usage: ..\\env\\Scripts\\python.exe scripts\\migrate_results_layout.py [--apply]
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, logging_utils

OLD_ROOT = config.RESULTS_DIR
METRIC_COLUMNS = ["rmse", "mae", "r2"]


def notebook_of(encoder: str) -> str:
    if encoder.startswith("E5_"):
        return "04"
    if encoder.startswith("E3_"):
        return "03"
    return "02" if encoder.endswith("_QUESTION") else "01"


def granularity_of(encoder: str) -> str:
    if encoder.startswith("E5_") or encoder.endswith("_QUESTION"):
        return config.GRANULARITY_QUESTION
    return config.GRANULARITY_PARTICIPANT


def family_of(encoder: str) -> str:
    return config.FAMILY_FINETUNING if encoder.startswith("E5_") else config.FAMILY_FROZEN


def split_log(path: Path, fields: list[str], filename: str, apply: bool) -> list[tuple]:
    """Splits one old log into the new tree. Returns (destination, rows)."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    buckets: dict[Path, list] = {}
    for row in rows:
        encoder = row["encoder"]
        row.setdefault("family", "") or None
        row["family"] = family_of(encoder)
        row["granularity"] = granularity_of(encoder)
        row["notebook"] = notebook_of(encoder)
        destination = config.results_dir(row["augmentation"], row["family"], row["granularity"])
        buckets.setdefault(destination, []).append(row)

    written = []
    for destination, bucket in buckets.items():
        target = destination / filename
        if apply:
            destination.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in bucket:
                    writer.writerow({k: row.get(k, "") for k in fields})
        written.append((target, bucket))
    return written


def move_embeddings(old_dir: Path, augmentation: str, apply: bool) -> list[tuple]:
    moves = []
    for path in sorted(old_dir.glob("*")):
        if not path.is_file():
            continue
        granularity = (config.GRANULARITY_QUESTION if "_questions" in path.name
                        else config.GRANULARITY_PARTICIPANT)
        destination = config.results_dir(augmentation, config.FAMILY_FROZEN,
                                          granularity) / "embeddings" / path.name
        moves.append((path, destination))
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
    return moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                         help="Perform the migration. Without it, only report what would move.")
    args = parser.parse_args()

    old_logs = [p for p in OLD_ROOT.glob("*/experiment_log.csv")] + \
               [p for p in OLD_ROOT.glob("*/*/experiment_log.csv")
                if p.parent.parent.name == "fine-tuning"]
    if not old_logs:
        print("Nothing to migrate: no logs found in the old layout.")
        return

    before = []
    for path in old_logs:
        frame = pd.read_csv(path)
        before.append(frame)
        print(f"found {len(frame):>4} rows in {path.relative_to(config.EXPERIMENT_ROOT)}")
    before = pd.concat(before, ignore_index=True)

    for path in old_logs:
        augmentation = path.parent.name if path.parent.parent == OLD_ROOT else path.parent.name
        for filename, fields in [(logging_utils.RUN_LOG_FILENAME, logging_utils.RUN_FIELDS),
                                  (logging_utils.TRIAL_LOG_FILENAME, logging_utils.TRIAL_FIELDS)]:
            source = path.parent / filename
            if not source.exists():
                continue
            for target, bucket in split_log(source, fields, filename, args.apply):
                print(f"  {'moved ' if args.apply else 'would move '}{len(bucket):>4} rows -> "
                      f"{target.relative_to(config.EXPERIMENT_ROOT)}")
        embeddings = path.parent / "embeddings"
        if embeddings.exists():
            moves = move_embeddings(embeddings, augmentation, args.apply)
            print(f"  {'moved ' if args.apply else 'would move '}{len(moves)} embedding files")

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform the migration.")
        return

    # Only the new tree: load_all_runs would also pick up the originals,
    # which are still on disk at this point, and double every count.
    migrated = [p for p in config.RESULTS_DIR.rglob(logging_utils.RUN_LOG_FILENAME)
                 if p.parent.parts[-3] in (config.FAMILY_FROZEN, config.FAMILY_FINETUNING)]
    after = pd.concat([pd.read_csv(p) for p in migrated], ignore_index=True)
    ok = len(before) == len(after)
    for column in METRIC_COLUMNS:
        ok &= sorted(before[column].round(6)) == sorted(after[column].round(6))
    print(f"\nverification: {len(before)} rows before, {len(after)} after, "
          f"metric values identical: {ok}")
    if ok:
        for path in old_logs:
            shutil.rmtree(path.parent, ignore_errors=True)
        print("old layout removed")
    else:
        print("MISMATCH: old layout left in place for inspection")


if __name__ == "__main__":
    main()
