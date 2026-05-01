import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import evaluate
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,

    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT-Video inference with BERTScore evaluation")
    parser.add_argument("--input", required=True, help="Path to input .pkl file")
    parser.add_argument("--output", required=True, help="Path to output .json file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--model-id", default="llava-hf/LLaVA-NeXT-Video-7B-hf", help="LLaVA model ID")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save output every N samples")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed samples in existing output file")
    parser.add_argument("--verbose", action="store_true", help="Show debug-level logs")
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    return model, processor


def build_prompt(processor, question):
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an advanced AI assistant designed to answer questions based on video content. "
                "Your task is to carefully analyze the video frames, paying close attention to both temporal and spatial information. "
                "Ensure that you understand the sequence of events and how they relate to the specific question asked. "
                "Your answers should be precise, directly addressing the question with relevant details extracted from the video. "
                "Provide contextually accurate responses by integrating insights from the entire video, and make sure your answers are concise and focused."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Watch the video and answer the following question: {question}"},
                {"type": "video"},
            ],
        },
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def run_inference(model, processor, sample_item, max_new_tokens):
    prompt = build_prompt(processor, sample_item["question"])
    clip = np.array(sample_item["video_frames"])
    inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.module.device)
    output = model.module.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True)


def score_answer(sentence_model, bertscorer, generated, reference):
    ref_embedding = sentence_model.encode([reference], show_progress_bar=False)
    gen_embedding = sentence_model.encode([generated], show_progress_bar=False)
    similarity_score = float(cosine_similarity(ref_embedding, gen_embedding)[0][0])
    bert_result = bertscorer.compute(
        predictions=[generated],
        references=[reference],
        model_type="roberta-large",
    )
    bert_score = float(bert_result["f1"][0])
    return similarity_score, bert_score


def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_results(output_path, results):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.hf_token:
        login(token=args.hf_token)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    logging.info(f"Loading data from {args.input}")
    sample = load_pickle(args.input)
    logging.info(f"Loaded {len(sample)} samples")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    results = {}
    if args.resume:
        results = load_existing_results(args.output)
        logging.info(f"Resuming — {len(results)} samples already done, skipping them")

    logging.info(f"Loading model: {args.model_id}")
    model, processor = build_model(args.model_id)
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    bertscorer = evaluate.load("bertscore")

    similarity_scores = []
    bert_scores = []
    pending = [i for i in range(1, len(sample) + 1) if i not in results]

    for idx, item in enumerate(tqdm(pending), start=1):
        try:
            generated_answer = run_inference(model, processor, sample[item], args.max_new_tokens)
            similarity_score, bert_score = score_answer(
                sentence_model, bertscorer, generated_answer, sample[item]["answer"]
            )

            similarity_scores.append(similarity_score)
            bert_scores.append(bert_score)

            results[item] = {
                "Question": sample[item]["question"],
                "Original Answer": sample[item]["answer"],
                "Generated Answer": generated_answer,
                "Similarity Score": similarity_score,
                "BERTScore": bert_score,
            }

            logging.debug(f"[{item}] sim={similarity_score:.4f} bert={bert_score:.4f} | {generated_answer[:80]}")

            if idx % args.checkpoint_every == 0:
                save_results(args.output, results)
                logging.debug(f"Checkpoint saved at sample {item}")

        except Exception as exc:
            logging.warning(f"Sample {item} failed: {exc}")

    save_results(args.output, results)

    if similarity_scores:
        logging.info(f"Average Similarity Score: {sum(similarity_scores) / len(similarity_scores):.4f}")
        logging.info(f"Average BERTScore:        {sum(bert_scores) / len(bert_scores):.4f}")
    logging.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()