"""Tokenize an instruction-tuning corpus once and save to disk as HuggingFace
Arrow datasets.

Single-config example (legacy Magicoder default):
    python scripts/data/prepare_data.py \
        --model_name allenai/OLMo-2-0425-1B \
        --max_seq_length 2048 \
        --max_train_samples 24000 \
        --max_eval_samples 512 \
        --seed 0 \
        --out_dir data/magicoder_seq2048

Multi-config example (opc-sft-stage2 all 4 sub-configs concat):
    python scripts/data/prepare_data.py \
        --model_name allenai/OLMo-2-0425-1B \
        --dataset_name OpenCoder-LLM/opc-sft-stage2 \
        --dataset_configs educational_instruct,evol_instruct,mceval_instruct,package_instruct \
        --max_seq_length 2048 \
        --max_train_samples 1000000 \
        --max_eval_samples 1024 \
        --eval_fraction 0.01 \
        --seed 0 \
        --out_dir data/opc_sft_stage2_all_packed_seq2048

Output schema depends on `--data_pipeline_version`:
  - "packed_v1" (default): train side emits per-doc rows with
    `input_ids` (variable length) + `prompt_len` (int); eval side same.
    Packing happens at train-time (in train.py), not here, so the same
    Arrow dataset can be reused across runs that differ in seq_length.
  - "unpacked_v0": legacy single-`input_ids` column, no boundary tracked.
"""
import argparse
import os

from datasets import concatenate_datasets, load_dataset

from lora_playground.train import (
    DATA_PIPELINE_VERSIONS,
    load_splits,
    pack_train_dataset,
    select_prefix,
    tokenize_splits,
    tokenize_splits_with_boundary,
)
from transformers import AutoTokenizer


def load_multi_config_splits(args):
    """Load multiple configs of the same dataset, concat their train splits,
    then carve an eval split via train_test_split. Mirrors the
    `lora_playground.train.load_splits` flow when no held-out test split
    is available.
    """
    configs = [c.strip() for c in args.dataset_configs.split(",") if c.strip()]
    print(f"  loading {len(configs)} configs of {args.dataset_name}:")
    per_config = []
    for cfg in configs:
        d = load_dataset(args.dataset_name, cfg, split=args.train_split)
        # Keep only columns that all configs share (instruction/output is the
        # working contract; extras like `tag`, `code`, `entry_point`,
        # `testcase`, `seq_id` differ across configs and would block concat).
        keep = {"instruction", "output", "input", "response"}
        drop = [c for c in d.column_names if c not in keep]
        if drop:
            d = d.remove_columns(drop)
        print(f"    {cfg}: {len(d):,} rows, columns={d.column_names}")
        per_config.append(d)
    full = concatenate_datasets(per_config)
    print(f"  concatenated: {len(full):,} rows")
    split = full.train_test_split(
        test_size=args.eval_fraction, seed=args.seed, shuffle=True,
    )
    train = split["train"]
    eval_ds = split["test"]
    train = select_prefix(train.shuffle(seed=args.seed), args.max_train_samples)
    eval_ds = select_prefix(eval_ds.shuffle(seed=args.seed + 1), args.max_eval_samples)
    return train, eval_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--dataset_name", default="ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response")
    parser.add_argument("--dataset_config", default=None,
                        help="Single config name. Mutually exclusive with --dataset_configs.")
    parser.add_argument("--dataset_configs", default=None,
                        help="Comma-separated list of configs to load and concatenate "
                             "(e.g. for OpenCoder-LLM/opc-sft-stage2 with all 4 sub-configs). "
                             "Train splits of each config are concatenated, then an eval "
                             "split is carved via train_test_split with --eval_fraction.")
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
    parser.add_argument("--tokenize_num_proc", type=int, default=None,
                        help="num_proc forwarded to dataset.map for tokenization. "
                             "Default single-process; set to e.g. 8 for parallel "
                             "tokenization on multi-core hosts.")
    parser.add_argument(
        "--data_pipeline_version",
        choices=DATA_PIPELINE_VERSIONS,
        default="packed_v1.1",
        help="Tokenization schema. packed_v1.1 (default) tokenizes per-doc, "
             "then runs the offline greedy pack (drops zero-supervision slots) "
             "so train.py reads slots directly. packed_v1 emits per-doc rows "
             "and defers packing to train startup (legacy; non-deterministic "
             "across seeds, wastes CPU per run). unpacked_v0 emits a "
             "single tokenized text column (legacy).",
    )
    args = parser.parse_args()

    if args.dataset_configs and args.dataset_config:
        parser.error("Pass either --dataset_config or --dataset_configs, not both.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset splits ...")
    if args.dataset_configs:
        train_raw, eval_raw = load_multi_config_splits(args)
    else:
        train_raw, eval_raw = load_splits(args)
    print(f"  train: {len(train_raw)} examples, eval: {len(eval_raw)} examples")

    print(f"Tokenizing (data_pipeline_version={args.data_pipeline_version}) ...")
    if args.data_pipeline_version.startswith("packed_v1"):
        train_tok, eval_tok = tokenize_splits_with_boundary(
            train_raw, eval_raw, tokenizer, args,
        )
    else:
        train_tok, eval_tok = tokenize_splits(
            train_raw, eval_raw, tokenizer, args,
        )
    print(f"  train tokens columns: {train_tok.column_names}")

    # packed_v1.1: pre-pack the train side here, so train.py reads slots
    # directly (no per-run startup re-pack, deterministic slot ordering
    # across seeds). pack_train_dataset applies the zero-supervision-slot
    # filter (packed_v1.1 default). Eval stays per-doc — PadToMaxCollator
    # pads each doc to seq_length at batch time.
    if args.data_pipeline_version == "packed_v1.1":
        n_docs = len(train_tok)
        print(f"Pre-packing train side ({n_docs} docs → slots @ seq={args.max_seq_length}) ...")
        train_tok = pack_train_dataset(
            train_tok,
            seq_length=args.max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        n_slots = len(train_tok)
        print(f"  packed: {n_docs} docs → {n_slots} slots "
              f"(mean {n_docs/max(n_slots,1):.2f} docs/slot)")
        print(f"  train slot columns: {train_tok.column_names}")

    train_out = os.path.join(args.out_dir, "train")
    eval_out = os.path.join(args.out_dir, "eval")
    print(f"Saving to {args.out_dir} ...")
    train_tok.save_to_disk(train_out)
    eval_tok.save_to_disk(eval_out)
    print(f"  saved {len(train_tok)} train examples → {train_out}")
    print(f"  saved {len(eval_tok)} eval examples  → {eval_out}")


if __name__ == "__main__":
    main()
