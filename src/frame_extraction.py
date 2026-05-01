"""
Frame extraction & data-loader builder for VidInstructQA.

This script supports the two video-QA benchmarks used in the paper:
  - NExT-QA (parquet metadata: question / answer / type / video)
  - MSVD-QA (JSON metadata: question / answer / file_name)

It performs three steps end-to-end:
  1. Walk the video directory, locate each clip referenced in the metadata,
     and uniformly sample N frames per video at 224x224.
  2. Persist intermediate per-batch pickles so an interrupted run can be
     resumed cheaply (each batch holds ~50 video samples by default).
  3. Optionally merge a contiguous range of batch pickles into a single
     dataloader-ready file consumed by `src/next_qa/inference.py`
     and `src/msvd_qa/inference.py`.

Output schema for every sample (matches what the inference scripts expect):
    {
        "video_frames": List[np.ndarray]   # H x W x 3, BGR (OpenCV default)
        "question":     str,
        "answer":       str,
        "type":         str,               # NExT-QA: CW/CH/TN/TC/TP/DC/DL/DO/DB
                                           # MSVD-QA: filled with first word of Q
                                           #          (What/Who/How/When/Where)
    }

Typical usage
-------------
# 1) Extract frames into per-batch pickles (NExT-QA, validation split, 4 frames)
python frame_extraction.py extract \
    --dataset nextqa \
    --metadata ../NExTQA/OE/validation-00000-of-00001.parquet \
    --video-dir ../NExTQA/videos/ \
    --split val \
    --n-frames 4 \
    --batch-size 50 \
    --out-dir ../extracted_feats

# 2) Merge batches 1..40 into a single dataloader pickle
python frame_extraction.py merge \
    --in-dir ../extracted_feats/val/4_frames/full_feats_val \
    --out-file ../data_loaders/val/4_frames/dataloader_val_4_frames_1.pkl \
    --start 1 --end 41

# 3) Quick visual sanity check on one sample of a dataloader pickle
python frame_extraction.py preview \
    --in-file ../data_loaders/val/4_frames/dataloader_val_4_frames_1.pkl \
    --sample-index 1
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
REQUIRED_SAMPLE_KEYS = {"video_frames", "question", "answer", "type"}


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level utilities
# ──────────────────────────────────────────────────────────────────────────────
def extract_frames(
    video_path: str,
    n_frames: int = 8,
    frame_size: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """
    Uniformly sample ``n_frames`` frames from a video and resize each to
    ``frame_size``. Returns a list of HxWx3 BGR uint8 arrays. If the video
    cannot be opened or contains no frames an empty list is returned.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[extract_frames] Could not open: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    step = max(1, total_frames // n_frames)
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, frame_size))

    cap.release()
    return frames


def find_video(video_id: str, video_dir: str) -> Optional[str]:
    """
    Walk ``video_dir`` recursively and return the first file whose name
    starts with ``video_id`` and ends with a known video extension.
    """
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.startswith(video_id) and fname.endswith(VIDEO_EXTENSIONS):
                return os.path.join(root, fname)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pickle I/O
# ──────────────────────────────────────────────────────────────────────────────
def save_pickle(obj: Any, path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_batch(
    video_data: Dict[Any, Dict[str, Any]],
    batch_number: int,
    n_frames: int,
    split: str,
    out_dir: str,
) -> str:
    save_dir = os.path.join(out_dir, split, f"{n_frames}_frames", f"full_feats_{split}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(save_dir, f"{batch_number}.pkl")
    save_pickle(video_data, filename)
    print(f"  Saved {filename}  ({len(video_data)} samples)")
    return filename


# ──────────────────────────────────────────────────────────────────────────────
# Metadata loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_nextqa_metadata(parquet_path: str) -> List[Dict[str, Any]]:
    """NExT-QA parquet has columns: video, question, answer, type, ..."""
    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]
    required = {"video", "question", "answer", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")
    return [
        {
            "video_id": str(row["video"]),
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
        }
        for _, row in df.iterrows()
    ]


def load_msvd_metadata(json_path: str) -> List[Dict[str, Any]]:
    """
    MSVD-QA JSON entries have: file_name, question, answer.
    The 'type' field is not stored in the original metadata, so we infer
    a coarse category from the question's first word so that downstream
    per-category scoring (What/Who/How/When/Where) still works.
    """
    df = pd.read_json(json_path)
    df.columns = [c.lower() for c in df.columns]
    required = {"file_name", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"JSON is missing required keys: {missing}")

    samples: List[Dict[str, Any]] = []
    valid_types = {"what", "who", "how", "when", "where"}
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        first = q.split()[0].lower() if q else ""
        qtype = first.capitalize() if first in valid_types else "Other"
        samples.append(
            {
                "video_id": str(row["file_name"]),
                "question": q,
                "answer": row["answer"],
                "type": qtype,
            }
        )
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: extract
# ──────────────────────────────────────────────────────────────────────────────
def run_extraction(
    samples: Iterable[Dict[str, Any]],
    video_dir: str,
    n_frames: int,
    split: str,
    out_dir: str,
    frame_size: Tuple[int, int] = (224, 224),
    batch_size: int = 50,
) -> None:
    video_frames_dict: Dict[Any, Dict[str, Any]] = {}
    batch_number = 1
    n_processed = 0
    n_missing = 0

    for sample in samples:
        vid = sample["video_id"]
        path = find_video(vid, video_dir)
        if path is None:
            n_missing += 1
            print(f"[extract] Video not found: {vid}")
            continue

        frames = extract_frames(path, n_frames=n_frames, frame_size=frame_size)
        if not frames:
            n_missing += 1
            print(f"[extract] No frames extracted: {vid}")
            continue

        # Use the (video_id, processed_index) as the key so questions sharing
        # the same video do not overwrite one another.
        key = f"{vid}_{n_processed}"
        video_frames_dict[key] = {
            "video_frames": frames,
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample["type"],
        }
        n_processed += 1

        if len(video_frames_dict) >= batch_size:
            save_batch(video_frames_dict, batch_number, n_frames, split, out_dir)
            video_frames_dict = {}
            batch_number += 1

    # Flush the remainder
    if video_frames_dict:
        save_batch(video_frames_dict, batch_number, n_frames, split, out_dir)

    print(
        f"\n[extract] Done. Processed: {n_processed} | "
        f"Missing/failed: {n_missing} | "
        f"Batches written: {batch_number - (1 if not video_frames_dict and n_processed % batch_size == 0 else 0)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: merge batches into a single dataloader pickle
# ──────────────────────────────────────────────────────────────────────────────
def is_valid_sample(sample: Any) -> bool:
    if not isinstance(sample, dict):
        return False
    if not REQUIRED_SAMPLE_KEYS.issubset(sample.keys()):
        return False
    q = sample.get("question")
    if q is None:
        return False
    if isinstance(q, str) and q.strip() == "":
        return False
    if hasattr(q, "__len__") and not isinstance(q, str) and len(q) == 0:
        return False
    frames = sample.get("video_frames")
    if frames is None or len(frames) == 0:
        return False
    return True


def merge_batches(
    in_dir: str,
    out_file: str,
    start: int,
    end: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Read batch pickles ``{in_dir}/{i}.pkl`` for i in [start, end) and merge
    every valid sample into a single integer-keyed dictionary written to
    ``out_file``. Existing keys in source pickles are discarded; the merged
    file is rekeyed with consecutive integers starting at 1, which is what
    the inference scripts iterate over.
    """
    merged: Dict[int, Dict[str, Any]] = {}
    counter = 1

    files_found = files_missing = 0
    raw_samples = valid_samples = invalid_samples = 0

    for file_num in range(start, end):
        path = os.path.join(in_dir, f"{file_num}.pkl")
        if not os.path.exists(path):
            files_missing += 1
            continue

        files_found += 1
        try:
            data = load_pickle(path)
        except Exception as exc:
            print(f"[merge] Could not read {path}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"[merge] {path} is not a dict, skipping")
            continue

        raw_samples += len(data)
        for _, sample in data.items():
            if not is_valid_sample(sample):
                invalid_samples += 1
                continue
            merged[counter] = {
                "video_frames": sample["video_frames"],
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample["type"],
            }
            counter += 1
            valid_samples += 1

    print("\n[merge] Summary")
    print(f"  Files found       : {files_found}")
    print(f"  Files missing     : {files_missing}")
    print(f"  Raw samples       : {raw_samples}")
    print(f"  Valid samples     : {valid_samples}")
    print(f"  Invalid samples   : {invalid_samples}")
    print(f"  Final keys        : {len(merged)}")

    save_pickle(merged, out_file)
    print(f"  Wrote merged file : {out_file}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: preview
# ──────────────────────────────────────────────────────────────────────────────
def preview_sample(in_file: str, sample_index: Optional[int] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is required for the 'preview' subcommand. "
            "Install it with `pip install matplotlib`."
        )

    data = load_pickle(in_file)
    print(f"Total samples in {in_file}: {len(data)}")

    if sample_index is None:
        sample_index = next(iter(data.keys()))
    sample = data[sample_index]

    frames = sample["video_frames"]
    print(f"\nQuestion: {sample['question']}")
    print(f"Answer  : {sample['answer']}")
    print(f"Type    : {sample['type']}")
    print(f"Frames  : {len(frames)}")

    n = len(frames)
    plt.figure(figsize=(3 * n, 3))
    for i, frame in enumerate(frames):
        plt.subplot(1, n, i + 1)
        # OpenCV gives BGR; matplotlib expects RGB
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Frame {i + 1}")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frame extraction and dataloader builder for VidInstructQA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Sample frames from videos and save per-batch pickles.")
    p_ext.add_argument("--dataset", choices=["nextqa", "msvd"], required=True)
    p_ext.add_argument(
        "--metadata",
        required=True,
        help="Path to NExT-QA parquet (train/val/test) or MSVD-QA JSON.",
    )
    p_ext.add_argument("--video-dir", required=True, help="Directory containing the raw video files.")
    p_ext.add_argument("--split", required=True, help="Split label used in the output path (train/val/test).")
    p_ext.add_argument("--n-frames", type=int, default=8, help="Frames per video (default: 8).")
    p_ext.add_argument("--batch-size", type=int, default=50, help="Samples per batch pickle (default: 50).")
    p_ext.add_argument("--frame-size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    p_ext.add_argument("--out-dir", default="../extracted_feats", help="Root output directory for batch pickles.")

    # merge
    p_mrg = sub.add_parser("merge", help="Merge a contiguous range of batch pickles into one dataloader file.")
    p_mrg.add_argument("--in-dir", required=True, help="Directory holding the per-batch {i}.pkl files.")
    p_mrg.add_argument("--out-file", required=True, help="Path of the merged dataloader pickle to write.")
    p_mrg.add_argument("--start", type=int, default=1, help="First batch index to include (inclusive).")
    p_mrg.add_argument("--end", type=int, required=True, help="Stop index (exclusive).")

    # preview
    p_prv = sub.add_parser("preview", help="Display a single sample from a merged dataloader pickle.")
    p_prv.add_argument("--in-file", required=True)
    p_prv.add_argument("--sample-index", type=int, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "extract":
        if args.dataset == "nextqa":
            samples = load_nextqa_metadata(args.metadata)
        else:
            samples = load_msvd_metadata(args.metadata)
        print(f"Loaded {len(samples)} metadata rows from {args.metadata}")

        run_extraction(
            samples=samples,
            video_dir=args.video_dir,
            n_frames=args.n_frames,
            split=args.split,
            out_dir=args.out_dir,
            frame_size=tuple(args.frame_size),
            batch_size=args.batch_size,
        )

    elif args.command == "merge":
        merge_batches(
            in_dir=args.in_dir,
            out_file=args.out_file,
            start=args.start,
            end=args.end,
        )

    elif args.command == "preview":
        preview_sample(args.in_file, sample_index=args.sample_index)


if __name__ == "__main__":
    main()
