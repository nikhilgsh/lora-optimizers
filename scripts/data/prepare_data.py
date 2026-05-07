"""Tokenize Magicoder once and save to disk as HuggingFace Arrow datasets.

Usage:
    python scripts/data/prepare_data.py \
        --model_name allenai/OLMo-2-0425-1B \
        --max_seq_length 512 \
        --max_train_samples 4096 \
        --max_eval_samples 512 \
        --seed 0 \
        --out_dir data/magicoder_seq512
"""
import argparse
import os

from lora_playground.train import load_splits, tokenize_splits
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--dataset_name", default="ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=4096)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset splits ...")
    train_raw, eval_raw = load_splits(args)
    print(f"  train: {len(train_raw)} examples, eval: {len(eval_raw)} examples")

    print("Tokenizing ...")
    train_tok, eval_tok = tokenize_splits(train_raw, eval_raw, tokenizer, args)
    print(f"  train tokens columns: {train_tok.column_names}")

    train_out = os.path.join(args.out_dir, "train")
    eval_out = os.path.join(args.out_dir, "eval")
    print(f"Saving to {args.out_dir} ...")
    train_tok.save_to_disk(train_out)
    eval_tok.save_to_disk(eval_out)
    print(f"  saved {len(train_tok)} train examples → {train_out}")
    print(f"  saved {len(eval_tok)} eval examples  → {eval_out}")


if __name__ == "__main__":
    main()
