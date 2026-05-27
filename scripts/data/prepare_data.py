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

Column-map example (meta-math/MetaMathQA, has {query, response}):
    python scripts/data/prepare_data.py \
        --model_name allenai/OLMo-2-0425-1B \
        --dataset_name meta-math/MetaMathQA \
        --column_map "query=instruction" \
        --max_seq_length 2048 --max_train_samples 1000000 --max_eval_samples 1024 \
        --eval_fraction 0.01 --seed 0 \
        --out_dir data/metamath_qa_packed_seq2048

Chat-flatten example (allenai/tulu-3-sft-mixture, has {messages: [...]}):
    python scripts/data/prepare_data.py \
        --model_name allenai/OLMo-2-0425-1B \
        --dataset_name allenai/tulu-3-sft-mixture \
        --chat_field messages \
        --max_seq_length 2048 --max_train_samples 400000 --max_eval_samples 1024 \
        --eval_fraction 0.01 --seed 0 \
        --out_dir data/tulu3_sft_mixture_400k_packed_seq2048
    # Note: --chat_field filter runs AFTER --max_train_samples; if it drops
    # >1% of rows you may want to raise --max_train_samples to compensate.

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


def parse_column_map(spec):
    """Parse "old=new,old2=new2" into {"old": "new", ...}. Empty/None → {}."""
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--column_map entry {pair!r} missing '='")
        old, new = pair.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def apply_column_map(ds, column_map):
    if not column_map:
        return ds
    present = [old for old in column_map if old in ds.column_names]
    if not present:
        return ds
    return ds.rename_columns({old: column_map[old] for old in present})


def flatten_chat_messages(messages, eos_token):
    """Take a list of {role, content} dicts and return (prompt, response)
    using the Tulu chat template (see allenai/open-instruct
    dataset_transformation.py:237 'tulu' template, lines 240–252).

    Format:
        <|system|>\\n{sys}\\n
        <|user|>\\n{u1}\\n
        <|assistant|>\\n{a1}{eos}\\n
        ...
        <|user|>\\n{uN}\\n
        <|assistant|>\\n          ← prompt ends here (add_generation_prompt=True)
        {response}                 ← response is the final assistant content
                                    (eos appended by the tokenizer at train time)

    Returns (None, None) if the doc has no trailing assistant turn — caller filters.
    """
    if not isinstance(messages, list) or not messages:
        return None, None
    # Validate trailing assistant turn (preserving original system position).
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return None, None
    response = (messages[-1].get("content") or "").strip()
    if not response:
        return None, None
    history = messages[:-1]
    if not history:
        return None, None
    pieces = []
    for t in history:
        if not isinstance(t, dict):
            continue
        role = t.get("role", "")
        content = (t.get("content") or "")
        if role == "system":
            pieces.append(f"<|system|>\n{content}\n")
        elif role == "user":
            pieces.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            pieces.append(f"<|assistant|>\n{content}{eos_token}\n")
        # silently drop other roles (tool, function, etc. — rare in Tulu-3 SFT)
    if not pieces:
        return None, None
    prompt = "".join(pieces) + "<|assistant|>\n"
    return prompt, response


def apply_chat_flatten(ds, chat_field, eos_token):
    """Transform a dataset with a chat list column into {prompt, response}
    using the Tulu chat template. Routes through train.py's existing
    `{prompt, response}` boundary (train.py:85), avoiding the
    Instruction:/Response: wrapper applied to the {instruction, output} path."""
    if chat_field is None:
        return ds
    if chat_field not in ds.column_names:
        raise ValueError(f"--chat_field {chat_field!r} not in dataset columns {ds.column_names}")

    def _map(example):
        prompt, response = flatten_chat_messages(example[chat_field], eos_token)
        return {"prompt": prompt, "response": response}

    ds = ds.map(_map, remove_columns=ds.column_names, desc="flattening chat (tulu template)")
    # Drop rows without a clean boundary (no trailing assistant turn / empty content).
    before = len(ds)
    ds = ds.filter(lambda ex: ex["prompt"] is not None and ex["response"] is not None,
                   desc="filtering empty chats")
    after = len(ds)
    if before != after:
        print(f"  chat-flatten dropped {before - after}/{before} rows missing trailing assistant turn")
    return ds


def load_multi_config_splits(args, eos_token):
    """Load multiple configs of the same dataset, concat their train splits,
    then carve an eval split via train_test_split. Mirrors the
    `lora_playground.train.load_splits` flow when no held-out test split
    is available.
    """
    configs = [c.strip() for c in args.dataset_configs.split(",") if c.strip()]
    print(f"  loading {len(configs)} configs of {args.dataset_name}:")
    column_map = parse_column_map(args.column_map)
    per_config = []
    for cfg in configs:
        d = load_dataset(args.dataset_name, cfg, split=args.train_split)
        d = apply_column_map(d, column_map)
        if args.chat_field:
            d = apply_chat_flatten(d, args.chat_field, eos_token)
        # Keep only columns that train.py's boundary code consumes (see train.py:75-102).
        # `prompt`/`response` for chat-flattened data, `instruction`/`output`/`input` for
        # the standard SFT schema. Extras like `tag`, `code`, `entry_point`, `testcase`,
        # `seq_id` differ across configs and would block concat.
        keep = {"instruction", "output", "input", "response", "prompt"}
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
    parser.add_argument("--column_map", default=None,
                        help='Comma-separated old=new column renames applied after load. '
                             'E.g. "query=instruction" for meta-math/MetaMathQA. '
                             'Unknown columns are silently skipped.')
    parser.add_argument("--chat_field", default=None,
                        help='Name of a list-of-{role, content} column to flatten '
                             'into {instruction, output}. E.g. "messages" for '
                             'allenai/tulu-3-sft-mixture. Drops rows without a '
                             'trailing assistant turn.')
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
        # multi-config path applies column_map / chat_field per-config before concat
        train_raw, eval_raw = load_multi_config_splits(args, tokenizer.eos_token)
    else:
        train_raw, eval_raw = load_splits(args)
        column_map = parse_column_map(args.column_map)
        train_raw = apply_column_map(train_raw, column_map)
        eval_raw = apply_column_map(eval_raw, column_map)
        if args.chat_field:
            train_raw = apply_chat_flatten(train_raw, args.chat_field, tokenizer.eos_token)
            eval_raw = apply_chat_flatten(eval_raw, args.chat_field, tokenizer.eos_token)
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
