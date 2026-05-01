#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Basic LLaVA inference on MSVD-QA (zero-shot, uniform sampling)")
    parser.add_argument("--video_dir", required=True, help="Directory containing MSVD-QA videos")
    parser.add_argument("--data_file", required=True, help="Path to JSON data file")
    parser.add_argument("--output", required=True, help="Path to output .json file")
    parser.add_argument("--split", default="val", help="Split name (for logging)")
    parser.add_argument("--n_frames", type=int, default=8, choices=[4, 8, 16, 32], help="Frames per video")
    parser.add_argument("--model-id", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="LLaVA model ID")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save output every N samples")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed samples in existing output file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    return parser.parse_args()


def build_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    elif torch.cuda.is_available():
        model = model.cuda()
        print("Using single GPU")
    else:
        print("Using CPU")
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    return model, processor


def find_video(video_id, video_dir):
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.startswith(str(video_id)) and file.endswith(ext):
                    return os.path.join(root, file)
    return None


def extract_frames_uniform(video_path, n_frames=8, frame_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
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


def build_prompt(processor, question):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Watch the video and answer the following question: {question}"},
                {"type": "video"},
            ],
        },
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def run_inference(model, processor, video_path, question, n_frames, max_new_tokens):
    frames = extract_frames_uniform(video_path, n_frames=n_frames)
    if not frames:
        return "[ERROR: No frames extracted]"
    prompt = build_prompt(processor, question)
    inputs = processor(text=prompt, videos=np.array(frames), padding=True, return_tensors="pt")
    device = model.module.device if hasattr(model, "module") else model.device
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)


def load_data(data_file):
    df = pd.read_json(data_file)
    df.to_parquet("output.parquet", engine="pyarrow", index=False)
    return df


def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {}


def save_results(output_path, results):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    print(f"Loading data from {args.data_file}")
    df = load_data(args.data_file)
    print(f"Processing {len(df)} samples from {args.split} split")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    results = {}
    if args.resume:
        results = load_existing_results(args.output)
        print(f"Resuming — {len(results)} samples already done, skipping them")

    videos_found = sum(1 for _, row in df.iterrows() if find_video(str(row["file_name"]), args.video_dir))
    print(f"Found {videos_found}/{len(df)} videos")
    if videos_found == 0:
        print(f"No videos found in {args.video_dir}")
        return

    print(f"Loading model: {args.model_id}")
    model, processor = build_model(args.model_id)

    processed = skipped = 0
    for step, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing"), start=1):
        video_id = str(row["file_name"])
        key = f"{video_id}_{idx}"
        if args.resume and key in results:
            continue
        video_path = find_video(video_id, args.video_dir)
        if not video_path:
            skipped += 1
            continue
        try:
            generated_answer = run_inference(
                model, processor, video_path, row["question"], args.n_frames, args.max_new_tokens
            )
            results[key] = {
                "Question": row["question"],
                "Original Answer": row["answer"],
                "Generated Answer": generated_answer,
            }
            processed += 1
            if step % args.checkpoint_every == 0:
                save_results(args.output, results)
        except Exception as exc:
            tqdm.write(f"WARNING: {video_id} failed: {exc}")
            skipped += 1

    save_results(args.output, results)
    print(f"Processed: {processed} | Skipped: {skipped}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()