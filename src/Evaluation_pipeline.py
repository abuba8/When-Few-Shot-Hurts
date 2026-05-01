import argparse
import json
import logging
import os

import torch
from huggingface_hub import login
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "Prompts")

TEMPLATES = {
    "zero": "LLM_as_judge_base.j2",
    "one":  "LLM_as_judge_base.j2",
    "few":  "LLM_as_judge_few.j2",
}

EXAMPLES_FILES = {
    "one": "examples/one_shot.json",
    "few": "examples/few_shot.json",
}


def build_instruction_prompt(shot_type: str, k: str) -> str:
    template_str = open(os.path.join(PROMPTS_DIR, TEMPLATES[shot_type])).read()
    template = Template(template_str)

    if shot_type == "zero":
        return template.render(shot_type=shot_type)

    examples_path = os.path.join(PROMPTS_DIR, EXAMPLES_FILES[shot_type])
    with open(examples_path) as f:
        data = json.load(f)

    return template.render(shot_type=shot_type, selected_k=k, data=data)


def load_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
        torch_dtype=torch.float16,
    )
    return tokenizer, model


def compare_answers(question: str, original_answer: str, generated_answer: str,
                    instruction_prompt: str, tokenizer, model) -> str:
    prompt = (
        f"{instruction_prompt}\n\n"
        f"**Question:** {question}\n"
        f"**Original Answer:** {original_answer}\n"
        f"**Generated Answer:** {generated_answer}\n"
        f"**Response:**"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip().split()[-2]


def parse_args():
    parser = argparse.ArgumentParser(description="Llama judge: evaluate semantic similarity of QA pairs")
    parser.add_argument("input_json", help="Path to input JSON file with QA pairs")
    parser.add_argument("output_json", help="Path to write output JSON with similarity scores")
    parser.add_argument(
        "--shot-type",
        choices=["zero", "one", "few"],
        default="few",
        help="Prompting strategy: zero, one, or few shot (default: few)",
    )
    parser.add_argument(
        "--k",
        default="K1",
        help="Example set to use from the examples file, e.g. K1, K2, K3 (default: K1)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B",
        help="HuggingFace model name (default: meta-llama/Meta-Llama-3.1-8B)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (defaults to $HF_TOKEN env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.hf_token:
        raise ValueError("HuggingFace token required: pass --hf-token or set $HF_TOKEN")

    login(token=args.hf_token)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    torch.cuda.empty_cache()

    instruction_prompt = build_instruction_prompt(args.shot_type, args.k)

    with open(args.input_json, "r") as f:
        qa_data = json.load(f)

    tokenizer, model = load_model(args.model)

    for idx in tqdm(qa_data.keys(), desc="Processing Questions", dynamic_ncols=True):
        entry = qa_data[idx]
        entry["Similarity"] = compare_answers(
            entry["Question"], entry["Original Answer"], entry["Generated Answer"],
            instruction_prompt, tokenizer, model,
        )

    with open(args.output_json, "w") as f:
        json.dump(qa_data, f, indent=4)

    print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()