"""
Accuracy aggregator for VidInstructQA judge outputs.

Loads one or more judge JSON files (the output of `Evaluation_pipeline.py`),
merges them, and reports overall accuracy plus a per-category breakdown that
matches the dataset's question-type taxonomy.

Each input JSON entry must carry a `Similarity` field whose value is "Yes"
(case-insensitive) for a correct prediction and anything else for incorrect.

Datasets
--------
nextqa : per-category breakdown uses the `type` column from the NExT-QA
         parquet (CW, CH, TN, TC, TP, DC, DL, DO, DB) grouped into
         Causal / Temporal / Descriptive. The parquet path is required.
msvd   : per-category breakdown is inferred from the question's first word
         (What / Who / How / When / Where). No parquet needed.

Examples
--------
# NExT-QA: overall + fine-grained + broad-category accuracy
python compute_accuracy.py nextqa results_run1.json results_run2.json \
    --parquet ../NExTQA/OE/validation-00000-of-00001.parquet

# MSVD-QA: overall + per-Wh-word breakdown
python compute_accuracy.py msvd results_msvd_val.json

# MSVD-QA: overall accuracy only (skip the breakdown)
python compute_accuracy.py msvd results_msvd_val.json --no-breakdown
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# NExT-QA type taxonomy
# ──────────────────────────────────────────────────────────────────────────────
NEXTQA_TYPE_TO_CATEGORY = {
    "CW": "Causal",
    "CH": "Causal",
    "TN": "Temporal",
    "TC": "Temporal",
    "TP": "Temporal",
    "DC": "Descriptive",
    "DL": "Descriptive",
    "DO": "Descriptive",
    "DB": "Descriptive",
}

NEXTQA_TYPE_FULL_NAMES = {
    "CW": "Causal (Why)",
    "CH": "Causal (How)",
    "TN": "Temporal (Next)",
    "TC": "Temporal (Current)",
    "TP": "Temporal (Previous)",
    "DC": "Descriptive (Count)",
    "DL": "Descriptive (Location)",
    "DO": "Descriptive (Other)",
    "DB": "Descriptive (Binary Yes/No)",
}

MSVD_CATEGORIES = ("What", "Who", "How", "When", "Where")


# ──────────────────────────────────────────────────────────────────────────────
# I/O + merging
# ──────────────────────────────────────────────────────────────────────────────
def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def combine_json_files(paths: List[str]) -> Dict[str, Any]:
    """Merge multiple judge JSONs. Colliding keys get a `_file{i}` suffix."""
    combined: Dict[str, Any] = {}
    for idx, path in enumerate(paths, start=1):
        for key, value in load_json(path).items():
            if key in combined:
                combined[f"{key}_file{idx}"] = value
            else:
                combined[key] = value
    return combined


def is_correct(similarity_value: Any) -> bool:
    return str(similarity_value).strip().lower() == "yes"


# ──────────────────────────────────────────────────────────────────────────────
# Question-type attachment
# ──────────────────────────────────────────────────────────────────────────────
def attach_nextqa_types(data: Dict[str, Any], parquet_path: str) -> Dict[str, Any]:
    """Look up each entry's NExT-QA type code by matching on the question text."""
    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]
    if {"question", "type"} - set(df.columns):
        raise ValueError(
            f"Parquet must contain 'question' and 'type' columns. Got: {list(df.columns)}"
        )

    question_to_type: Dict[str, str] = {}
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        question_to_type.setdefault(q, row["type"])

    matched = unmatched = 0
    for entry in data.values():
        q = entry.get("Question", "").strip()
        if q in question_to_type:
            entry["_type"] = question_to_type[q]
            matched += 1
        elif entry.get("Type"):
            entry["_type"] = entry["Type"]
            matched += 1
        else:
            entry["_type"] = "UNKNOWN"
            unmatched += 1

    print(f"  Question-type matching: {matched} matched | {unmatched} unmatched")
    return data


def msvd_category_from_question(entry: Dict[str, Any]) -> Optional[str]:
    q = str(entry.get("Question", "")).strip()
    if not q:
        return None
    first = q.split()[0].lower()
    return first.capitalize() if first.capitalize() in MSVD_CATEGORIES else None


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def _print_overall(total: int, correct: int, header: str) -> float:
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    acc = correct / total if total else 0.0
    print(f"\nTotal questions  : {total}")
    print(f"Correct (Yes)    : {correct}")
    print(f"Incorrect        : {total - correct}")
    print(f"Overall Accuracy : {acc:.2%}")
    return acc


def report_nextqa(data: Dict[str, Any]) -> Tuple[float, Dict, Dict]:
    total = correct = 0
    fine: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    broad: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for entry in data.values():
        total += 1
        ok = is_correct(entry.get("Similarity", ""))
        correct += int(ok)

        t = entry.get("_type", "UNKNOWN")
        fine[t]["total"] += 1
        fine[t]["correct"] += int(ok)

        cat = NEXTQA_TYPE_TO_CATEGORY.get(t, "Other")
        broad[cat]["total"] += 1
        broad[cat]["correct"] += int(ok)

    overall = _print_overall(total, correct, "RESULTS SUMMARY (NExT-QA)")

    print("\n" + "-" * 60)
    print("Accuracy by BROAD CATEGORY")
    print("-" * 60)
    print(f"  {'Category':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
    for cat in ("Causal", "Temporal", "Descriptive", "Other"):
        if cat in broad:
            s = broad[cat]
            acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"  {cat:<14} {s['correct']:>8} {s['total']:>8} {acc:>10.2%}")

    print("\n" + "-" * 60)
    print("Accuracy by FINE-GRAINED TYPE")
    print("-" * 60)
    print(f"  {'Code':<6} {'Full Name':<28} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*8} {'-'*10}")
    for code in sorted(fine.keys(), key=lambda t: (NEXTQA_TYPE_TO_CATEGORY.get(t, "ZZZ"), t)):
        s = fine[code]
        acc = s["correct"] / s["total"] if s["total"] else 0
        name = NEXTQA_TYPE_FULL_NAMES.get(code, code)
        print(f"  {code:<6} {name:<28} {s['correct']:>8} {s['total']:>8} {acc:>10.2%}")

    return overall, dict(broad), dict(fine)


def report_msvd(data: Dict[str, Any], breakdown: bool = True) -> Tuple[float, Dict]:
    total = correct = 0
    cat_stats: Dict[str, Dict[str, int]] = {c: {"total": 0, "correct": 0} for c in MSVD_CATEGORIES}

    for entry in data.values():
        total += 1
        ok = is_correct(entry.get("Similarity", ""))
        correct += int(ok)

        if breakdown:
            cat = msvd_category_from_question(entry)
            if cat:
                cat_stats[cat]["total"] += 1
                cat_stats[cat]["correct"] += int(ok)

    overall = _print_overall(total, correct, "RESULTS SUMMARY (MSVD-QA)")

    if breakdown:
        print("\n" + "-" * 60)
        print("PER-CATEGORY ACCURACY (inferred from question's first word)")
        print("-" * 60)
        print(f"  {'Category':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
        for cat in MSVD_CATEGORIES:
            s = cat_stats[cat]
            acc = s["correct"] / s["total"] if s["total"] else 0
            print(f"  {cat:<10} {s['correct']:>8} {s['total']:>8} {acc:>10.2%}")

    return overall, cat_stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute overall and per-category accuracy from VidInstructQA judge outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dataset",
        choices=["nextqa", "msvd"],
        help="Which dataset taxonomy to use for the per-category breakdown.",
    )
    p.add_argument(
        "json_files",
        nargs="+",
        help="One or more judge JSON files (output of Evaluation_pipeline.py).",
    )
    p.add_argument(
        "--parquet",
        default=None,
        help="NExT-QA parquet (with 'question' and 'type' columns). Required for --dataset nextqa.",
    )
    p.add_argument(
        "--no-breakdown",
        action="store_true",
        help="Print overall accuracy only (skip per-category breakdown).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    for f in args.json_files:
        if not Path(f).exists():
            raise SystemExit(f"ERROR: input file not found: {f}")

    print(f"\nLoading {len(args.json_files)} JSON file(s)...")
    data = combine_json_files(args.json_files)
    print(f"  Combined entries: {len(data)}")

    if args.dataset == "nextqa":
        if args.no_breakdown:
            total = len(data)
            correct = sum(1 for e in data.values() if is_correct(e.get("Similarity", "")))
            _print_overall(total, correct, "RESULTS SUMMARY (NExT-QA)")
            return
        if not args.parquet:
            raise SystemExit("ERROR: --parquet is required for --dataset nextqa.")
        if not Path(args.parquet).exists():
            raise SystemExit(f"ERROR: parquet file not found: {args.parquet}")
        print(f"\nAttaching question types from: {args.parquet}")
        data = attach_nextqa_types(data, args.parquet)
        report_nextqa(data)
    else:
        report_msvd(data, breakdown=not args.no_breakdown)


if __name__ == "__main__":
    main()
