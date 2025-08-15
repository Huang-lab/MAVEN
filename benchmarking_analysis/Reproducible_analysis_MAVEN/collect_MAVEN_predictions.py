#!/usr/bin/env python3
"""
Create two subsets of MAVEN prediction files based on metadata CSVs.

Usage:
    python collect_maven_subsets.py <source_s12_dir> <destination_root_dir>

Example:
    python collect_maven_subsets.py ../../MAVEN_test/s12-proteome-wide-predictions resource_data/
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
from typing import Iterable, Set, Tuple

def read_unique_uniprot_ids(csv_path: Path, column_name: str, split_before_underscore: bool = False) -> Set[str]:
    """Read a CSV and return unique UniProt IDs from the given column.
    If split_before_underscore=True, take the substring before the first underscore."""
    ids: Set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column_name not in (reader.fieldnames or []):
            raise ValueError(
                f'Column "{column_name}" not found in {csv_path}. '
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(column_name) or "").strip()
            if not raw:
                continue
            uid = raw.split("_", 1)[0].strip() if split_before_underscore else raw
            if uid:
                ids.add(uid)
    return ids

def expected_filename(uniprot_id: str) -> str:
    return f"{uniprot_id}_maven.tsv.gz"

def collect_existing_files(s12_dir: Path, ids: Iterable[str]) -> Tuple[Set[Path], Set[str]]:
    """Return (paths that exist, ids missing)."""
    existing: Set[Path] = set()
    missing: Set[str] = set()
    for uid in ids:
        path = s12_dir / expected_filename(uid)
        if path.exists():
            existing.add(path)
        else:
            missing.add(uid)
    return existing, missing

def copy_files(files: Iterable[Path], dest_dir: Path, overwrite: bool = False, dry_run: bool = False) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in files:
        dst = dest_dir / src.name
        if dst.exists() and not overwrite:
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Subset MAVEN s12 predictions using metadata CSVs.")
    parser.add_argument("source_s12_dir", type=Path,
                        help="Path to the source s12 folder containing [UNIPROT]_maven.tsv.gz files.")
    parser.add_argument("destination_root", type=Path,
                        help="Path to the root folder where output folders will be created.")
    parser.add_argument("--scores-csv", type=Path, default=Path("mave_performance_benchmark_dataset_scores.csv"),
                        help='Path to mave_performance_benchmark_dataset_scores.csv (uses column "Uniprot_id").')
    parser.add_argument("--dms-csv", type=Path, default=Path("DMS_substitutions.csv"),
                        help='Path to DMS_substitutions.csv (uses column "UniProt_ID" and splits before underscore).')
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in destination folders if present.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without copying.")
    args = parser.parse_args()

    # Destination folder paths
    subset_dir = args.destination_root / "s12-proteome-wide-predictions-maven-benchmarking"
    maven_predictions_dir = args.destination_root / "proteingym_data" / "MAVEN_predictions"

    # Sanity checks
    if not args.source_s12_dir.exists() or not args.source_s12_dir.is_dir():
        raise SystemExit(f"ERROR: Source s12 directory not found: {args.source_s12_dir}")
    if not args.scores_csv.exists():
        raise SystemExit(f"ERROR: scores CSV not found: {args.scores_csv}")
    if not args.dms_csv.exists():
        raise SystemExit(f"ERROR: DMS CSV not found: {args.dms_csv}")

    # 1) Subset from benchmark scores
    ids_scores = read_unique_uniprot_ids(args.scores_csv, "Uniprot_id", split_before_underscore=False)
    files_scores, missing_scores = collect_existing_files(args.source_s12_dir, ids_scores)
    copied_scores = copy_files(files_scores, subset_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    print(f"[1] Benchmark scores subset")
    print(f"    Unique UniProt IDs found in CSV: {len(ids_scores)}")
    print(f"    Matching files present in source: {len(files_scores)}")
    print(f"    Files {'to copy' if args.dry_run else 'copied'} to {subset_dir}: {copied_scores}")
    if missing_scores:
        print(f"    Missing in source (no matching file): {len(missing_scores)}")

    # 2) MAVEN_predictions subset
    ids_dms = read_unique_uniprot_ids(args.dms_csv, "UniProt_ID", split_before_underscore=True)
    files_dms, missing_dms = collect_existing_files(args.source_s12_dir, ids_dms)
    copied_dms = copy_files(files_dms, maven_predictions_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    print(f"[2] MAVEN_predictions subset")
    print(f"    Unique UniProt IDs found in CSV: {len(ids_dms)}")
    print(f"    Matching files present in source: {len(files_dms)}")
    print(f"    Files {'to copy' if args.dry_run else 'copied'} to {maven_predictions_dir}: {copied_dms}")
    if missing_dms:
        print(f"    Missing in source (no matching file): {len(missing_dms)}")

    if args.dry_run:
        print("\n(DRY RUN) No files were actually copied. Remove --dry-run to perform the copy.")

if __name__ == "__main__":
    main()
