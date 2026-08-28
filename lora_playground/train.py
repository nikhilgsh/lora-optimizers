import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)

from .checkpoint import (
    ckpt_dir_for_step,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
)
from .data import PackingCollator, PadToMaxCollator, pack_documents
from .distributed import (
    all_reduce_mean,
    cleanup as dist_cleanup,
    get_rank,
    get_world_size,
    init_distributed,
    is_main,
)
from .mfu import (
    compute_mfu,
    count_total_params,
    device_peak_tflops,
    flops_per_token_for_mode,
)
from .optim import (
    MSIGN_CHOICES,
    OPTIMIZER_CHOICES,
    PRECOND_CHOICES,
    build_optimizer,
    optimizer_config_dict,
    optimizer_effective_config,
)
from .publication_semantics import build_optimizer_variant_semantics_payload
from .publication_identity import LORA_INIT_B_CHOICES
from .run_schema import (
    ATTEMPT_ID_ENV,
    CHECKPOINT_IDENTITY_ENV,
    attempt_metadata,
    semantic_revisions,
)
from .training_kernel import (
    batch_to_device,
    build_peft_model,
    count_tokens,
    run_one_train_step,
)


TRAINING_MODES = ("lora", "svd_step_oracle", "svd_cumulative_oracle", "galore", "ucv")
DATA_PIPELINE_VERSIONS = ("packed_v1.1", "packed_v1", "unpacked_v0")


def _current_attempt_metadata(args) -> dict:
    """Resolve one execution attempt without inferring a resume parent.

    Submission tooling supplies stable per-task checkpoint identity and a new
    attempt ID on every launch. Direct invocations still get explicit IDs: a
    checkpoint/resume path is a stable local lineage namespace, while a run
    that cannot resume uses its own attempt as the namespace. The actual
    parent is filled only after ``load_checkpoint`` succeeds and is emitted in
    the resume event, never guessed here from the presence of ``--resume_from``.
    """
    attempt_id = os.environ.get(ATTEMPT_ID_ENV) or f"local-{uuid.uuid4().hex}"
    checkpoint_identity = os.environ.get(CHECKPOINT_IDENTITY_ENV)
    if not checkpoint_identity:
        checkpoint_path = args.checkpoint_dir or args.resume_from
        checkpoint_identity = (
            f"local-checkpoint:{Path(checkpoint_path).resolve()}"
            if checkpoint_path
            else f"nonresumable:{attempt_id}"
        )
    return attempt_metadata(
        attempt_id=attempt_id,
        checkpoint_identity=checkpoint_identity,
    )


def _resume_replays_original_dataloader(args) -> bool:
    return bool(
        getattr(args, "resume_replay_original_dataloader", False)
        or getattr(args, "resume_debug_replay", False)
    )


def _resume_restores_rng_state(args) -> bool:
    return bool(getattr(args, "resume_debug_replay", False))


def format_example_with_boundary(example):
    """Return (prompt_text, response_text). Mirrors `format_example` but
    keeps the prompt/response boundary explicit so the tokenizer can
    record `prompt_len` for prompt-masking under packed_v1."""
    if "prompt" in example and "completion" in example:
        return example["prompt"].strip() + "\n", example["completion"].strip()
    if "instruction" in example and "response" in example:
        return (
            f"Instruction:\n{example['instruction']}\n\nResponse:\n",
            example["response"],
        )
    if "prompt" in example and "response" in example:
        return example["prompt"].strip() + "\n", example["response"].strip()
    if "instruction" in example and "output" in example:
        pieces = [f"Instruction:\n{example['instruction']}"]
        inp = example.get("input")
        if isinstance(inp, str) and inp.strip():
            pieces.append(f"Input:\n{inp}")
        prompt = "\n\n".join(pieces) + "\n\nResponse:\n"
        return prompt, example["output"]
    # No clean boundary available — treat whole text as response (no prompt
    # mask). Matches the LM-style loss; downstream cfg records this as a
    # "no_boundary" case so analysis can flag it.
    if "prompt" in example and isinstance(example["prompt"], str):
        return "", example["prompt"]
    for key in ("text", "content", "code"):
        if key in example and isinstance(example[key], str):
            return "", example[key]
    return "", "\n".join(f"{k}: {v}" for k, v in example.items() if isinstance(v, str))


def format_example(example):
    if "prompt" in example and "completion" in example:
        return f"{example['prompt'].strip()}\n{example['completion'].strip()}"
    if "instruction" in example and "response" in example:
        return f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
    if "prompt" in example and "response" in example:
        return f"{example['prompt'].strip()}\n{example['response'].strip()}"
    if "prompt" in example and isinstance(example["prompt"], str):
        return example["prompt"]
    if "instruction" in example and "output" in example:
        pieces = [f"Instruction:\n{example['instruction']}"]
        inp = example.get("input")
        if isinstance(inp, str) and inp.strip():
            pieces.append(f"Input:\n{inp}")
        pieces.append(f"Response:\n{example['output']}")
        return "\n\n".join(pieces)
    for key in ("text", "content", "code"):
        if key in example and isinstance(example[key], str):
            return example[key]
    return "\n".join(f"{k}: {v}" for k, v in example.items() if isinstance(v, str))


def parse_target_modules(value):
    if value == "all-linear":
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def select_prefix(dataset, limit):
    if limit is None or limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.select(range(limit))


def load_splits(args):
    if args.train_file:
        data_files = {"train": args.train_file}
        if args.eval_file:
            data_files[args.eval_split] = args.eval_file
        raw = load_dataset("json", data_files=data_files)
    else:
        raw = load_dataset(args.dataset_name, args.dataset_config) if args.dataset_config else load_dataset(args.dataset_name)
    if not isinstance(raw, DatasetDict):
        raise ValueError("Expected load_dataset without split to return a DatasetDict.")

    if args.train_split not in raw:
        raise ValueError(f"Train split '{args.train_split}' not found. Available splits: {list(raw)}")

    train = raw[args.train_split]
    if args.eval_split in raw:
        eval_dataset = raw[args.eval_split]
    elif "validation" in raw:
        eval_dataset = raw["validation"]
    elif "test" in raw:
        eval_dataset = raw["test"]
    else:
        split = train.train_test_split(test_size=args.eval_fraction, seed=args.seed, shuffle=True)
        train = split["train"]
        eval_dataset = split["test"]

    train = select_prefix(train.shuffle(seed=args.seed), args.max_train_samples)
    eval_dataset = select_prefix(eval_dataset.shuffle(seed=args.seed + 1), args.max_eval_samples)
    return train, eval_dataset


def tokenize_splits(train, eval_dataset, tokenizer, args):
    """Legacy unpacked_v0 tokenization. Emits a single `input_ids` column;
    no prompt/response boundary tracked. Kept for back-compat with runs at
    `--data_pipeline_version unpacked_v0`."""
    def add_text(batch):
        keys = list(batch.keys())
        rows = [
            {key: batch[key][index] for key in keys}
            for index in range(len(batch[keys[0]]))
        ]
        return {"sample_text": [format_example(row) for row in rows]}

    def tokenize(batch):
        return tokenizer(
            batch["sample_text"],
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

    train_text = train.map(add_text, batched=True, remove_columns=train.column_names, desc="Formatting train")
    eval_text = eval_dataset.map(add_text, batched=True, remove_columns=eval_dataset.column_names, desc="Formatting eval")
    train_tok = train_text.map(tokenize, batched=True, remove_columns=train_text.column_names, desc="Tokenizing train")
    eval_tok = eval_text.map(tokenize, batched=True, remove_columns=eval_text.column_names, desc="Tokenizing eval")
    return train_tok, eval_tok


def tokenize_splits_with_boundary(train, eval_dataset, tokenizer, args):
    """packed_v1 tokenization. Emits `input_ids` (variable length, no
    padding) plus `prompt_len` (int, # tokens to mask in `labels` so the
    loss only sees response positions). Truncation is applied to the
    response side; if the prompt alone exceeds `max_seq_length` the
    document is dropped (rare on Magicoder; logged at end)."""

    def tok(batch):
        keys = list(batch.keys())
        rows = [
            {key: batch[key][index] for key in keys}
            for index in range(len(batch[keys[0]]))
        ]
        out_ids: list[list[int]] = []
        out_pl: list[int] = []
        for row in rows:
            prompt, response = format_example_with_boundary(row)
            prompt_ids = tokenizer(
                prompt, add_special_tokens=True
            )["input_ids"] if prompt else []
            response_ids = tokenizer(
                response, add_special_tokens=False
            )["input_ids"] if response else []
            full = prompt_ids + response_ids
            if len(full) > args.max_seq_length:
                # Truncate from the response side. If even the prompt is
                # too long, the doc has 0 response tokens and contributes
                # no supervised objective.
                full = full[: args.max_seq_length]
            prompt_len = min(len(prompt_ids), len(full))
            if len(full) == 0 or prompt_len >= len(full):
                continue
            out_ids.append(full)
            out_pl.append(prompt_len)
        return {"input_ids": out_ids, "prompt_len": out_pl}

    num_proc = getattr(args, "tokenize_num_proc", None) or None
    train_tok = train.map(
        tok, batched=True, remove_columns=train.column_names,
        desc="Tokenizing train (with boundary)", num_proc=num_proc,
    )
    eval_tok = eval_dataset.map(
        tok, batched=True, remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval (with boundary)", num_proc=num_proc,
    )
    return train_tok, eval_tok


def pack_train_dataset(train_tok, seq_length: int, pad_token_id: int):
    """Convert a per-doc tokenized HF Dataset into packed-slot rows.

    Pulls input_ids + prompt_len for every doc, runs greedy first-fit
    packing, returns a HF Dataset where each row is one packed slot
    (input_ids, labels, position_ids, doc_lens, all length seq_length
    except doc_lens which is variable).
    """
    from datasets import Dataset
    docs = [
        {"input_ids": row["input_ids"], "prompt_len": int(row["prompt_len"])}
        for row in train_tok
    ]
    slots = pack_documents(docs, seq_length=seq_length, pad_token_id=pad_token_id)
    return Dataset.from_list(slots)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Held-out NLL. Under DDP, each rank evaluates its DistributedSampler
    shard; the per-rank (loss-sum, token-count) pair is then all-reduced so
    every rank returns the same global weighted mean."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        tokens = count_tokens(batch)
        outputs = model(**batch)
        total_loss += float(outputs.loss.detach()) * tokens
        total_tokens += tokens
    model.train()
    # Pack into tensors so all_reduce_mean can do the global aggregation.
    loss_t = torch.tensor(total_loss, device=device, dtype=torch.float64)
    toks_t = torch.tensor(total_tokens, device=device, dtype=torch.float64)
    return all_reduce_mean(loss_t, toks_t)


def attach_heldout_factor_grads(model, optimizer, batch, device):
    """Attach token-weighted held-out gradients without changing the train step.

    The optimizer reads the transient gradients during its shadow diagnostic.
    Train gradients, model mode, and random-number-generator state are restored
    exactly before the optimizer step.
    """
    pairs = getattr(optimizer, "pairs", None)
    if not pairs:
        raise ValueError("held-out optimizer probe requires LoRA factor pairs")
    batches = [batch] if isinstance(batch, dict) else list(batch)
    heldout_batches = [batch_to_device(b, device) for b in batches]
    batch_tokens = [count_tokens(b) for b in heldout_batches]
    if not heldout_batches or any(n == 0 for n in batch_tokens):
        raise ValueError("held-out optimizer probe batch has no supervised tokens")
    total_tokens = sum(batch_tokens)
    saved_grads = [(A.grad.detach().clone(), B.grad.detach().clone())
                   for A, B in pairs]
    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if device.type == "cuda" and torch.cuda.is_available() else None)
    was_training = model.training
    try:
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        for heldout_batch, n_tokens in zip(heldout_batches, batch_tokens):
            weighted_loss = model(**heldout_batch).loss * (n_tokens / total_tokens)
            weighted_loss.backward()
        optimizer._heldout_factor_grads = [
            (A.grad.detach().float().clone(), B.grad.detach().float().clone())
            for A, B in pairs
        ]
    finally:
        optimizer.zero_grad(set_to_none=True)
        for (A, B), (gA, gB) in zip(pairs, saved_grads):
            A.grad = gA
            B.grad = gB
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def measure_heldout_factor_directions(
    model, optimizer, batch, device, *, identity_scale=None,
):
    """Evaluate complete shadow factor steps on held-out probe batches.

    The optimizer has already applied the lagged direction.  Its heavy
    diagnostic retains the pre-step factors and the three candidate factor
    directions, so each candidate can be evaluated at the actual step size and
    the applied parameters can then be restored exactly.  A single batch dict
    remains accepted; an iterable is aggregated with the same token weighting
    as :func:`evaluate`.
    """
    pairs = getattr(optimizer, "pairs", None)
    directions = getattr(optimizer, "_last_cw_heldout_directions", None)
    if not pairs or directions is None or len(directions) != len(pairs):
        raise ValueError("held-out loss probe is missing complete shadow directions")
    batches = [batch] if isinstance(batch, dict) else list(batch)
    heldout_batches = [batch_to_device(b, device) for b in batches]
    batch_tokens = [count_tokens(b) for b in heldout_batches]
    if not heldout_batches or any(n == 0 for n in batch_tokens):
        raise ValueError("held-out loss probe batch has no supervised tokens")
    post_factors = [(A.detach().clone(), B.detach().clone()) for A, B in pairs]
    cpu_rng = torch.get_rng_state()
    cuda_rng = (torch.cuda.get_rng_state_all()
                if device.type == "cuda" and torch.cuda.is_available() else None)
    was_training = model.training

    direction_key_sets = [
        set(record) - {"A_pre", "B_pre"} for record in directions.values()
    ]
    if any(keys != direction_key_sets[0] for keys in direction_key_sets[1:]):
        raise ValueError("held-out shadow direction labels differ across pairs")
    preferred_order = (
        "lagged", "fresh", "identity",
        "small_slot_uncentered", "small_slot_centered",
    )
    direction_labels = [
        label for label in preferred_order if label in direction_key_sets[0]
    ]
    direction_labels.extend(sorted(direction_key_sets[0] - set(direction_labels)))
    labels = ["pre", *direction_labels]
    if identity_scale is not None:
        if not math.isfinite(identity_scale) or identity_scale <= 0:
            raise ValueError("identity_scale must be finite and positive")
        labels.append("identity_scaled")

    def set_factors(label):
        for i, (A, B) in enumerate(pairs):
            record = directions[i]
            A_pre = record["A_pre"].to(dtype=A.dtype, device=A.device)
            B_pre = record["B_pre"].to(dtype=B.dtype, device=B.device)
            if label == "pre":
                A.copy_(A_pre)
                B.copy_(B_pre)
            else:
                direction_label = "identity" if label == "identity_scaled" else label
                dA, dB = record[direction_label]
                scale = identity_scale if label == "identity_scaled" else 1.0
                A.copy_(A_pre + scale * dA.to(dtype=A.dtype, device=A.device))
                B.copy_(B_pre + scale * dB.to(dtype=B.dtype, device=B.device))

    losses = {}
    losses_by_batch = {}
    try:
        model.eval()
        for label in labels:
            set_factors(label)
            batch_losses = [float(model(**b).loss.detach()) for b in heldout_batches]
            losses_by_batch[label] = batch_losses
            losses[label] = sum(
                loss * n for loss, n in zip(batch_losses, batch_tokens)
            ) / sum(batch_tokens)
        set_factors("pre")
        repeated_pre_batch_losses = [
            float(model(**b).loss.detach()) for b in heldout_batches
        ]
        repeated_pre_loss = sum(
            loss * n for loss, n in zip(repeated_pre_batch_losses, batch_tokens)
        ) / sum(batch_tokens)
    finally:
        for (A, B), (A_post, B_post) in zip(pairs, post_factors):
            A.copy_(A_post)
            B.copy_(B_post)
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        optimizer._last_cw_heldout_directions = {}

    result = {f"heldout_loss_{label}": loss for label, loss in losses.items()}
    result["heldout_probe_batches"] = len(heldout_batches)
    result["heldout_loss_pre_repeat"] = repeated_pre_loss
    result["heldout_loss_pre_repeat_abs_diff"] = abs(
        repeated_pre_loss - losses["pre"])
    result.update({
        f"heldout_loss_change_{label}": losses[label] - losses["pre"]
        for label in labels if label != "pre"
    })
    result.update({
        f"heldout_loss_change_{label}_per_batch": [
            loss - pre_loss
            for loss, pre_loss in zip(
                losses_by_batch[label], losses_by_batch["pre"])
        ]
        for label in labels if label != "pre"
    })
    return result


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _optimizer_provenance_fields(
    *,
    optimizer_name: str,
    optimizer,
    semantic_revision: str | int,
    implementation_revision: str | int,
) -> dict:
    """Build the three optimizer config-event blocks from one snapshot."""
    config = optimizer_config_dict(optimizer)
    effective = optimizer_effective_config(optimizer)
    return {
        "optimizer_config": config,
        "optimizer_effective": effective,
        "optimizer_variant_semantics": (
            build_optimizer_variant_semantics_payload(
                optimizer=optimizer_name,
                optimizer_instance=optimizer,
                optimizer_config=config,
                optimizer_effective=effective,
                semantic_revision=semantic_revision,
                implementation_revision=implementation_revision,
            )
        ),
    }


def git_dirty_state() -> dict:
    """Record plain ``git status --short`` output as audit provenance."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return {"git_dirty": None, "git_status": []}
    status = [line for line in result.stdout.splitlines() if line]
    return {"git_dirty": bool(status), "git_status": status}


def log_event(payload):
    """Emit JSON-line config/eval/train_step events. Rank-gated under DDP so
    only rank 0 writes — the on-disk log stays single-stream, matching
    single-GPU behavior."""
    if not is_main():
        return
    print(json.dumps(payload, sort_keys=True), flush=True)


def make_parser():
    parser = argparse.ArgumentParser(description="Lean LoRA optimizer training loop")
    parser.add_argument("--training_mode", choices=TRAINING_MODES, default="lora")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZER_CHOICES), default="adamw")
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--dataset_name", default="ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--data_dir", default=None, help="Pre-tokenized dataset dir (Arrow). Skips download + tokenization.")
    parser.add_argument(
        "--data_pipeline_version",
        choices=DATA_PIPELINE_VERSIONS,
        default="packed_v1.1",
        help="Data pipeline. 'packed_v1.1' (current default, 2026-05-14): "
             "same as packed_v1 but drops zero-supervision packed slots at "
             "pack time (slots whose labels are all -100 would produce NaN "
             "cross-entropy means and pollute Adam moments). 'packed_v1': "
             "train side packs tokenized docs into static seq_length slots "
             "with doc-aware SDPA mask + per-doc position_ids reset, eval "
             "pads each doc to seq_length; prompt-masked loss (labels=-100 "
             "on prompt). 'unpacked_v0': legacy DataCollatorForLanguageModeling "
             "path (dynamic shapes, no prompt mask, no doc-aware attention). "
             "All pre-2026-05-08 logs are unpacked_v0; runs 2026-05-08..14 "
             "are packed_v1; new runs default to packed_v1.1. Boundary "
             "recorded in cfg event so the loader can filter by version. "
             "See docs/notes/polar_product/data_pipeline_followups.md.",
    )
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--allow_multi_epoch",
        action="store_true",
        help="Bypass the single-pass guard. Use only for explicit "
             "multi-epoch experiments (most optimizer comparisons should not).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="Eval cadence in steps. Project convention: ALWAYS 200, "
             "regardless of --max_steps. Long-horizon runs get more eval "
             "points, not coarser ones — keeps trajectory granularity "
             "consistent across horizons so the same step-2000 eval can "
             "be compared between runs of different total length.",
    )
    parser.add_argument(
        "--train_loss_every",
        type=int,
        default=10,
        help="Per-step train-loss logging cadence (steps). 0 disables. "
             "Each emit is a `train_step` JSONL event with a windowed mean "
             "of step_loss over the last train_loss_every steps. Cheaper "
             "than eval (no held-out forward), so the cadence can be much "
             "tighter — useful for spotting magnitude-rule effects on "
             "training dynamics that average out at eval cadence.",
    )
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-rank micro batch size. Under DDP, the global "
                             "effective batch is batch_size × grad_accum_steps × "
                             "world_size. Use --global_batch_size to derive this "
                             "automatically from a single number.")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Per-rank gradient-accumulation micro-steps. Under "
                             "DDP this is per-rank; see --global_batch_size.")
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="If set, derives per-rank --batch_size and "
                             "--grad_accum_steps so that batch_size × "
                             "grad_accum_steps × world_size == global_batch_size. "
                             "Per-rank batch_size keeps its CLI value (used as "
                             "the memory-fitting micro-batch); grad_accum_steps "
                             "is overridden to match. Errors if the global value "
                             "doesn't divide cleanly.")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", default="constant")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--svd_rank", type=int, default=None)
    parser.add_argument("--svd_niter", type=int, default=4,
                        help="Power iterations for randomized SVD (svd oracle modes). "
                             "Use -1 for exact economy SVD (slow).")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_init_b",
        choices=LORA_INIT_B_CHOICES,
        default="zero",
        help=(
            "LoRA init scheme. 'zero' (default, PEFT standard): A Kaiming, B=0. "
            "'gaussian' (Init[B]): A=0, B~N(0,1/in_features). "
            "'symmetric' (Init[AB]): A keeps PEFT default; B sampled at A's std; "
            "PiSSA-style residual subtracts (alpha/r)*B0@A0 from base_layer so "
            "the merged weight at step 0 equals the pretrained weight."
        ),
    )
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--lora_plus_multiplier", type=float, default=1.0)
    parser.add_argument("--scaled_metric", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2",
                                 "flash_attention_4", "auto"],
                        help="Attention kernel for the base model. Default "
                             "is `sdpa` because packed_v1's varlen path is "
                             "incompatible with flash_attention_2's cu_seqlens "
                             "shape contract; opt in to flash_attention_2 only "
                             "for unpacked_v0 runs. "
                             "flash_attention_2 matches Schulman/Biderman recipes "
                             "and is ~1.5-2x faster on Llama/Qwen than sdpa. "
                             "flash_attention_4 (CuTeDSL) targets Hopper (sm_90) "
                             "and Blackwell (sm_100+); requires transformers >=5 "
                             "and the flash-attn-4 package (or kernels-hub fallback). "
                             "'auto' picks FA4 on sm_90+, FA2 on sm_80, sdpa otherwise. "
                             "Falls back through FA2 then sdpa if the requested "
                             "backend is unavailable.")
    parser.add_argument("--use_liger", action="store_true",
                        help="Apply LinkedIn Liger Kernel patches (RMSNorm, MLP, "
                             "RoPE, fused cross-entropy) to the base model BEFORE "
                             "PEFT wrap. Reported 1.1-1.3x E2E speedup + 30%% "
                             "memory savings on Llama-3.x. Requires liger-kernel "
                             "package; family-dispatched from model_name.")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--profile_dir", default="runs/profiles")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--target_eval_loss", type=float, default=None)
    parser.add_argument(
        "--abort_on_nan_eval", action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), terminate the run on the first eval where "
             "eval_loss is non-finite (NaN/Inf). Avoids burning GPU on dead "
             "runs that have gone permanently non-finite. Disable with "
             "--no-abort_on_nan_eval if you want to keep training through "
             "transient NaNs (e.g. debugging recovery dynamics).")
    parser.add_argument(
        "--abort_on_eval_loss_above", type=float, default=None,
        help="If set, terminate the run on the first eval where finite "
             "eval_loss exceeds this threshold. Catches catastrophic "
             "divergence that stays finite (e.g. loss explodes to 5+ without "
             "going NaN). Composes with --abort_on_nan_eval. Default: None "
             "(disabled). Pick a value well above the legitimate loss "
             "cluster to avoid aborting transiently-bad-but-recoverable runs.")
    parser.add_argument("--precond_gamma", type=float, default=0.5,
                        help="Fractional power for PSI-LoRA/KFAC-LoRA K-FAC scaling.")
    parser.add_argument("--precond_ema_beta", type=float, default=0.99,
                        help="EMA smoothing for PSI-LoRA/KFAC-LoRA K-FAC statistics.")
    parser.add_argument("--precond_delta", type=float, default=1e-6,
                        help="Damping floor for PSI-LoRA/KFAC-LoRA K-FAC statistics.")
    parser.add_argument("--curvature_whitening", action="store_true",
                        help="Chord-tight: whiten by an EMA of the factor-gradient "
                             "second moment (BᵀHB) instead of the geometric Gram (BᵀB).")
    parser.add_argument("--curvature_beta", type=float, default=0.99,
                        help="EMA decay for the curvature whitening metric "
                             "(matches SOAP tuned β_shampoo=0.99).")
    parser.add_argument("--precond_delta_relative", action="store_true",
                        help="σ_max-relative damping: replace absolute δ in "
                             "(S+δI)^{-1/2} with δ·σ_max(S). Currently honored "
                             "by the chord-tight family only. See "
                             "docs/notes/polar_product/init_damping_math.md §5.3.")
    parser.add_argument("--psi_inner_iters", type=int, default=1,
                        help="K, number of LoRSUM ALS iterations per PSI-LoRA step.")
    parser.add_argument("--psi_momentum", type=float, default=0.9,
                        help="α₁ momentum coefficient in PSI-LoRA Algorithm 3.")
    parser.add_argument("--psi_rho", type=float, default=0.01,
                        help="ρ proximal regularizer for PSI-LoRA F-LoRSUM.")
    parser.add_argument("--psi_momentum_rank", type=int, default=None,
                        help="Rank r_m for PSI-LoRA low-rank momentum (default: lora_r).")
    parser.add_argument("--muon_ns_steps", type=int, default=5,
                        help="Newton-Schulz iterations for Muon-family optimizers; 0 disables NS (Tier-2 sanity).")
    parser.add_argument("--ns_form", type=str, default="gram",
                        choices=["rect", "gram", "gram-norestart"],
                        help="Newton-Schulz kernel form for the clean polar pipeline. "
                             "'gram' (default) uses _newton_schulz_gram_batched (Dao 2026 "
                             "Algorithm 3, fp16+restart at τ=2) — 21% step-wall reduction "
                             "vs rect k=3 at 8B/r=256, validated end-to-end. "
                             "'rect' is the legacy path (_newton_schulz_batched on (r,d)); "
                             "keep available for trajectory comparisons against pre-gram sweeps. "
                             "'gram-norestart' is gram without the restart hedge "
                             "(validated on Tier 1 + tight-damping corpus for "
                             "cubic Muon at NS=5 — see tests/test_ns_gram.py). Only "
                             "consulted by magnitude_rule=spectral_chord_tight_clean.")
    parser.add_argument("--galore_update_proj_gap", type=int, default=200,
                        help="Steps between GaLore projection updates.")
    parser.add_argument("--galore_scale", type=float, default=0.25,
                        help="GaLore update scale factor.")
    parser.add_argument("--log_basic_diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Cheap per-step probes: norms, sat_frac, cond(S_A), cond(S_B), "
                             "adam_gauge_residual, lambda_dir_gain, cross-coupling magnitudes. "
                             "~2%% wall. ON by default. Disable with --no-log_basic_diagnostics "
                             "for absolute throughput.")
    parser.add_argument("--log_heavy_diagnostics", action=argparse.BooleanOptionalAction, default=False,
                        help="Expensive probes: direct-SVD chord_slack cross-check, higham "
                             "accuracy reference (extra eigh), power-iter accuracy probes, "
                             "Picard contraction/oscillation. ~10x wall at r=64. OFF by "
                             "default; enable only for mechanism-investigation sweeps.")
    parser.add_argument("--optim_diagnostics_every", type=int, default=20,
                        help="Cadence (in optimizer steps) for both --log_basic_diagnostics and "
                             "--log_heavy_diagnostics.")
    parser.add_argument(
        "--optim_heldout_probe",
        action="store_true",
        help="Diagnostic only: on the heavy-diagnostics cadence, evaluate each "
             "factorwise shadow direction against one disjoint eval-batch gradient.",
    )
    parser.add_argument(
        "--optim_heldout_probe_batches",
        type=int,
        default=1,
        help="Number of held-out batches over which to average the exact shadow "
             "loss (default: 1). The first batch also supplies the linear probe gradient.",
    )
    parser.add_argument(
        "--optim_heldout_probe_exit",
        action="store_true",
        help="Diagnostic only: exit immediately after writing the held-out shadow "
             "event, without scheduler, eval, or checkpoint work.",
    )
    parser.add_argument(
        "--optim_heldout_identity_scale",
        type=float,
        default=None,
        help="Diagnostic only: also evaluate the identity-slot shadow direction "
             "multiplied by this fixed scale. This does not alter the optimizer step.",
    )
    parser.add_argument(
        "--optim_small_slot_microbatch_probe",
        action="store_true",
        help="Diagnostic only: capture the existing training microbatch backwards "
             "and compare direct uncentered versus centered P_A/Q_B small-slot "
             "moments on the held-out factorwise shadow. No optimizer state is changed.",
    )
    parser.add_argument("--debug_higham_residual", action="store_true",
                        help="Debug-only: every higham `_spd_inv_half` call emits a JSONL "
                             "`higham_residual` event with ‖Z H Z − I‖_F per matrix and "
                             "presence of non-finite output. Used to diagnose higham failures "
                             "(NaN at high r, drift) post-mortem. Cheap (~5%% wall on diagnostic "
                             "cadence). Off by default.")
    parser.add_argument("--log_non_finite", action="store_true",
                        help="Emit per-step `non_finite_detected` (top-of-step "
                             "per-pair A/B/grad isfinite check) and "
                             "`non_finite_intermediate` (end-of-step chain "
                             "check across u_A/SA^{-1/2}/X_A/P_A/geo_A/dA/...) "
                             "events for the polar-product family. Identifies "
                             "WHICH pair / WHICH chain intermediate first goes "
                             "non-finite. ~10%% wall overhead at r=256 from "
                             "the ~448+~20*N isfinite kernel launches per step. "
                             "Default OFF; turn on for NaN-debugging runs.")
    parser.add_argument("--log_non_finite_start_step", type=int, default=1,
                        help="When --log_non_finite is enabled, skip its "
                             "isfinite scans before this optimizer step. "
                             "Useful for near-window NaN tracing without "
                             "paying diagnostic overhead for early steps. "
                             "Default 1 preserves existing behavior.")
    parser.add_argument("--debug_optimizer_state", action="store_true",
                        help="Emit verbose per-pair optimizer scalar telemetry "
                             "for the polar-product batched path. Debug-only; "
                             "one JSON event per shape group per selected step.")
    parser.add_argument("--debug_optimizer_state_every", type=int, default=1,
                        help="Step cadence for --debug_optimizer_state.")
    parser.add_argument("--debug_optimizer_state_start_step", type=int, default=1,
                        help="When --debug_optimizer_state is enabled, skip "
                             "verbose optimizer-state events before this "
                             "optimizer step. Default 1 preserves existing "
                             "behavior.")
    parser.add_argument("--debug_snapshot_dir", default=None,
                        help="Directory for optimizer non-finite .pt snapshots. "
                             "When set, the polar-product optimizer saves the "
                             "offending pair's tensors on a non-finite chain event.")
    parser.add_argument("--debug_snapshot_limit", type=int, default=8,
                        help="Maximum optimizer debug snapshots to write per run.")
    parser.add_argument("--debug_abort_on_non_finite", action="store_true",
                        help="After writing optimizer debug snapshots, raise on "
                             "the first non-finite optimizer intermediate instead "
                             "of continuing a dead run.")
    parser.add_argument("--precond_refresh_every", type=int, default=1,
                        help="K-step cadence for refreshing the per-pair Gram-preconditioner cache "
                             "(adam-scaled-lora, adam-lin-lora, adam-lin-core-lora, "
                             "adam-polar-product-lora, adamuon-polar-product-lora). K=1 reproduces "
                             "the original per-step behavior; K>1 reuses the cached preconditioner "
                             "for K-1 steps after each refresh, trading a small amount of staleness "
                             "for a large step-time speedup at high LoRA rank. "
                             "NO EFFECT on the curvature-whiten family (kl-diag-*, kl-shampoo-*, "
                             "diag-shampoo-*, curvature-whiten-*) UNLESS --precond_method=eigh: "
                             "there it is only the QR-eigenbasis refresh cadence, and the production "
                             "'gram_ns' path (like 'higham') rebuilds S^{-1/2} from the current Gram "
                             "every step, so K is inert. Do not read K>1 in one of those configs as "
                             "meaning the run used a stale preconditioner.")
    parser.add_argument("--precond_method", choices=["eigh", "higham", "gram_ns"], default=None,
                        help="Method for computing S^{-1/2} in the curvature/polar-product optimizers. "
                             "DEFAULT None = use each optimizer family's own default (curvature-whiten → "
                             "'eigh' QR eigenbasis; polar-product → 'higham' Newton-Schulz). Set explicitly "
                             "to override: 'higham' = coupled Iannazzo NS (matmul-only, avoids the eigh "
                             "kernel-launch storm at high rank); 'gram_ns' = Polar-Express Gram NS "
                             "(curvature-whiten only — eigh-free, fresh every step, no stale eigenbasis; "
                             "see docs/notes/inverse_sqrt_variant_plan.md); 'eigh' = reference eigendecomp. "
                             "Passing it explicitly for curvature-whiten now reaches the optimizer (it used "
                             "to be silently dropped by the spec skip).")
    parser.add_argument("--higham_iters", type=int, default=10,
                        help="Newton-Schulz iterations when --precond_method=higham. "
                             "10 is needed for κ ≈ 200 (the worst case observed for SB "
                             "during training); 5 is fine on well-conditioned SA only.")
    parser.add_argument("--higham_compute_dtype", choices=["fp32", "fp16"],
                        default="fp32",
                        help="Inner-iteration dtype for Higham. fp32 is the validated "
                             "default; fp16 runs n_iters-1 iters on tensor cores and "
                             "polishes with 1 fp32 iter (variant B from "
                             "`scripts/bench/bench_higham_variants.py`). At r=256 on "
                             "Blackwell this gives 2.16× kernel speedup with ~1e-2 "
                             "rel-err vs fp32 at production cond ranges. Not worth it "
                             "at r ≤ 64 (cast overhead exceeds matmul win).")
    parser.add_argument("--picard_alpha", type=float, default=1.0,
                        help="Damping on the Picard cross-coupling correction in "
                             "AdamPolarProductLoRA (only takes effect when picard_iters > 1). "
                             "α=1 standard Picard; α=0 zeros the cross-term; intermediate "
                             "values continuously interpolate between block-diagonal and "
                             "joint-NE targets.")
    parser.add_argument("--htmuon_p", type=float, default=None,
                        help="HTMuon σ → σ^p sub-mode of "
                             "adam-polar-product-lora-coupled-spectral-chord-tight-clean. "
                             "When set ∈ (0, 1], the polar output is left-multiplied by "
                             "(X X^T)^(p/2) so the singular-value transfer becomes σ → σ^p "
                             "instead of σ → 1. Use power-of-two reciprocals (0.5, 0.25, "
                             "0.125, 0.0625) for exact iterated-sqrt landing. None disables "
                             "the path (bit-identical to clean NS5).")
    parser.add_argument("--picard_iters_override", type=int, default=None,
                        help="Override picard_iters for AdamPolarProductLoRA "
                             "(adam-polar-product-lora-coupled). Default uses the "
                             "factory's hardcoded value (3 for coupled).")
    parser.add_argument("--cw_picard_iters", type=int, default=1,
                        help="Picard block-coordinate depth for the CurvatureWhitenLoRA "
                             "kl family (kl-shampoo[-polar]-lora, kl-diag[-polar]-lora). "
                             "k=1 (default) is the single-block step; k>=2 adds the "
                             "diagonal cross-coupling correction (kl_shampoo_polar_"
                             "derivation.md section Cross-coupling).")
    parser.add_argument("--rdinv_variant", choices=["A", "B", "VN"], default="A",
                        help="Reference scale for the relative-damping floor in the "
                             "CurvatureWhitenLoRA large-side diagonal metric (_rdinv). "
                             "'A' (default) = own op-norm (x/x_max+δ)^{-1/2}, the shipped "
                             "paper protagonist; 'B' = raw/unbiased KL gauge "
                             "(x+δ·x_max)^{-1/2} (same op-norm floor); 'VN' = von Neumann / "
                             "matrix Adafactor (x+δ·Tr(partner))^{-1/2}, trace-scaled. "
                             "δ is op-norm-relative for A/B but trace-relative for VN. "
                             "Only 'A' reproduces the paper figures.")
    parser.add_argument("--rdinv_delta", type=float, default=None,
                        help="Decoupled damping floor for the _rdinv (P,Q diagonal) "
                             "metric. Default None = use --precond_delta (coupled, which "
                             "also floors the small-side C_A/C_B inverse-sqrt). Set it to "
                             "sweep the diagonal floor alone (e.g. VN trace-δ) while "
                             "holding the curvature-inverse floor at --precond_delta.")
    parser.add_argument("--dump_pre_polar_dir", default=None,
                        help="Directory for the pre-polar (H) dump: the whitened-momentum "
                             "matrices msign is applied to (zA/zB at the _polar_ns_guarded "
                             "call sites). Consumed offline by "
                             "lora_playground.lmo_diagnostics to score cheap substitutes "
                             "for msign (REG row/column scaling, RACS two-sided scaling, "
                             "K-step PolarExpress) against the exact spectral-LMO optimum. "
                             "Put this on shared storage (/mnt/ceph/users/<user>/...), not "
                             "in the repo: the tensors are model-sized. Required when "
                             "--dump_pre_polar_every > 0.")
    parser.add_argument("--dump_pre_polar_every", type=int, default=0,
                        help="Step cadence for the pre-polar dump. 0 (default) = OFF. "
                             "Diagnostic only — the update is bit-identical either way.")
    parser.add_argument("--dump_pre_polar_pairs", default=None,
                        help="Comma-separated substrings of LoRA module names to dump "
                             "(e.g. 'layers.0.self_attn.q_proj,layers.15.mlp.down_proj'). "
                             "Default: an evenly-spaced stride of --dump_pre_polar_max_pairs "
                             "pairs, which spans early/late layers and both factor shapes.")
    parser.add_argument("--dump_pre_polar_max_pairs", type=int, default=6,
                        help="How many pairs the default evenly-spaced stride selects. "
                             "Ignored when --dump_pre_polar_pairs is given.")
    parser.add_argument("--cw_metric_init", default="1e-12",
                        help="Init of the CurvatureWhitenLoRA diagonal metric EMAs Q and P. "
                             "DEFAULT '1e-12': a FLOAT ε → P₀=Q₀=εI, the "
                             "branch-free prior-free init (ε ≪ the ~1e-7 curvature scale, so no "
                             "_rdinv step-one branch and no warmup bias; validated to reproduce "
                             "zero-init to ≤5e-4 across all 4 models). Other values are ablations: "
                             "'zero' (legacy; relies on the step-one branch), 'ones' (strong "
                             "identity prior, hurts +0.019), 'delta' (=δ damping floor, also hurts "
                             "since δ≫ curvature). All give an identical step-1 update.")
    parser.add_argument("--cw_nesterov", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use Nesterov-lookahead momentum (ĝ + β₁·m, Muon "
                             "convention) into the CurvatureWhitenLoRA whiten→polar "
                             "core instead of plain bias-corrected EMA. Ablation for "
                             "diag-shampoo[-polar]-lora; the operator-norm rescale "
                             "makes this a direction-only change.")
    parser.add_argument("--cw_no_radius", action="store_true",
                        help="LEGACY ablation (−adaptive-radius): ρ=lr (flat) instead of "
                             "ρ=lr/(σmax(A)+σmax(B)). The σ_max(W) pin is KEPT — this does NOT "
                             "remove magnitude control; the magnitude-rule ablation is "
                             "--cw_unpinned. Retired from the paper (2026-06-12), kept dormant.")
    parser.add_argument("--precond", choices=sorted(PRECOND_CHOICES),
                        default=None,
                        help="What fills the two rank-side slots (C_B, C_A) in the "
                             "curvature-whiten family. product: C_B=B^T P B, C_A=A Q A^T "
                             "(PoLoRA). one-sided: C_B=C_A=I EVERYWHERE -- in the p,q "
                             "updates as well as the direction, so "
                             "qhat=(1/r)diag(G_A^T G_A), phat=(1/r)diag(G_B G_B^T) and "
                             "dA=msign(Mhat_A Q^-1/2)Q^-1/2. factorwise: C_B=P_A, C_A=Q_B, "
                             "two persistent r x r EMAs fitted from the factor gradients "
                             "(P_A <- b2 P_A + (1-b2)/d_in G_A Q^-1 G_A^T, mirror for Q_B). "
                             "factorwise-diag: the same fitted EMAs restricted to their "
                             "r diagonal entries, with elementwise inverse roots. "
                             "All four share one (P,Q), one set of p,q updates and the same "
                             "rho=eta/(smax(A)+smax(B)) rule. Default None = inherit the "
                             "optimizer's own setting (product for kl-diag*, factorwise for "
                             "kl-shampoo*).")
    parser.add_argument("--msign", choices=sorted(MSIGN_CHOICES),
                        default="full",
                        help="How accurately the matrix sign is applied to the whitened "
                             "momenta Z_A=C_B^-1/2 Mhat_A Q^-1/2, Z_B=P^-1/2 Mhat_B C_A^-1/2. "
                             "ORTHOGONAL to --precond: that picks the curvature structure, "
                             "this picks how much spectral mixing happens inside the LMO "
                             "direction. full: U=msign(Z). diag: approximate the Gram inside "
                             "the matrix sign by its diagonal, giving "
                             "U_A=Diag(diag(Z_A Z_A^T))^-1/2 Z_A = rownorm(Z_A) and "
                             "U_B=Z_B Diag(diag(Z_B^T Z_B))^-1/2 = colnorm(Z_B) -- no r x r "
                             "matmul or inverse sqrt (RACS-style). With --precond one-sided "
                             "the whole direction is O(rd) rather than O(r^2 d). "
                             "Requires an optimizer that applies a matrix sign at all.")
    parser.add_argument("--cw_no_diag_curv", action="store_true",
                        help="ABLATION (−curvature): force the input/output diagonal "
                             "curvatures to identity → C_A=BᵀB, C_B=AAᵀ (partner-Gram, "
                             "iMuon-like). Tests whether the two-sided diagonal Shampoo "
                             "curvature helps. Requires the diag_metric (protagonist) path.")
    parser.add_argument("--cw_unpinned", action="store_true",
                        help="ABLATION (−pin / the LoRA-Muon step): remove the "
                             "operator-norm magnitude rule — true-scale inverse-sqrt + no "
                             "σ_max(W) rescale, apply dX=−η·W raw. With --cw_no_diag_curv this "
                             "is the bare partner-Gram decoupled sandwich (LoRA-Muon Alg 1). "
                             "UNSTABLE at B=0 — run with --lora_init_b symmetric. Requires "
                             "--precond_method gram_ns.")
    parser.add_argument("--cw_solved_rho", action="store_true",
                        help="SOLVED magnitude rule (GPT-opt polora_attn solved_rho port): "
                             "size ρ by the positive root of ρ·t+ρ²=η with t=‖B·U_A+U_B·A‖₂ "
                             "measured, instead of the bound ρ=η/(σmax(A)+σmax(B)). Keeps the "
                             "certificate ‖Δ(BA)‖₂≤η while spending the full budget. "
                             "Incompatible with --cw_unpinned/--cw_no_radius.")
    parser.add_argument("--cw_factor_a", type=float, default=0.0,
                        help="Per-factor shape exponent for A: c_A=(r/d_in)^a, folded "
                             "into the operator-norm radius (product cap preserved). "
                             "0=current (equal radius). Rules: Keller a=0, MuP a=0.5, "
                             "Codex-rowspace a=0.25.")
    parser.add_argument("--cw_factor_b", type=float, default=0.0,
                        help="Per-factor shape exponent for B: c_B=(d_out/r)^b. "
                             "0=current. Keller/MuP/Codex all use b=0.5 (expansion side).")
    parser.add_argument("--polar_core_remix_alpha", type=float, default=0.0,
                        help="Experimental core-coordinate remix coefficient. "
                             "0 disables it. Nonzero values replace the row(A) / "
                             "col(B) projections of (u_A, u_B) with remixed "
                             "versions before the Picard / polar pipeline.")
    parser.add_argument("--anderson_m", type=int, default=0,
                        help="Anderson(m) acceleration depth for the Picard inner "
                             "loop in adam-polar-product-lora-coupled. m=0 disables "
                             "(plain Picard, default). m>=1 keeps the last m (input, "
                             "output) iterates and mixes via Type-II Anderson; "
                             "expect 3-5 effective iters to converge in regimes "
                             "where plain Picard needs 16.")
    parser.add_argument("--anderson_reg", type=float, default=1e-10,
                        help="Tikhonov regularizer on the Anderson LSQ Gram matrix.")
    parser.add_argument("--soap_beta", type=float, default=0.95,
                        help="EMA factor for the r×r covariance matrices L_A=EMA(gA gA^T), "
                             "R_B=EMA(gB^T gB) used by adam-soap-polar-product-lora. "
                             "0.95 is the SOAP-paper default.")
    parser.add_argument("--soap_refresh_every", type=int, default=1,
                        help="Cadence (in steps) for re-eigendecomposing L_A, R_B to refresh "
                             "the SOAP eigenbases Q_A, Q_B in adam-soap-polar-product-lora. "
                             "Q stays at identity until the first refresh, so the first "
                             "soap_refresh_every-1 steps reduce exactly to "
                             "adam-polar-product-lora.")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam β₁ (momentum). Default 0.9.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam β₂ (variance EMA). Default 0.999. β₂=0 disables EMA "
                             "(instant per-step variance — tests whether EMA of v matters "
                             "for polar-pipeline upstream).")
    parser.add_argument("--polar_method", type=str, default="ns",
                        choices=["ns", "ns_hybrid", "polar_express", "ssc"],
                        help="Polar approximation method in adam-polar-product-lora's _polar_pipeline. "
                             "'ns' = standard degree-3 Newton-Schulz (default). "
                             "'ns_hybrid' = DeepSeek-V4 §2.4 two-stage degree-5 (8 aggressive + 2 refine). "
                             "'polar_express' = Amsel et al. arXiv:2505.16932 per-iteration optimal degree-5. "
                             "'ssc' = SPECTRA (arXiv:2603.14315) soft spectral clipping h_c(σ)=σ/√(1+(σ/c)²). "
                             "Requires --ssc_c.")
    parser.add_argument("--ssc_c", type=float, default=None,
                        help="SSC clipping threshold c (σ-units). Mutually exclusive with --ssc_kappa; "
                             "exactly one is required when --polar_method ssc. "
                             "Input to _ssc_misr_batched is post-§2.5-rescale (σ_max≈1), so c is in "
                             "fraction-of-σ_max units.")
    parser.add_argument("--ssc_kappa", type=float, default=None,
                        help="SSC κ-adaptive target rank-normalized energy "
                             "(1/r) Σ_i (h_c(s_i)/h_c(1))². Mutually exclusive with --ssc_c. "
                             "c is solved per-pair per-step via bisection. "
                             "Achievable range: (‖s‖²/r, 1] on the §2.5-rescaled spectrum; "
                             "concentrated LoRA polar inputs imply useful κ ≳ 0.1.")
    parser.add_argument("--ssc_nsteps", type=int, default=10,
                        help="MISR (matrix iterative soft-clipping) iteration count for SSC. "
                             "Default 10 matches the production-validated SSC κ-adaptive runs; "
                             "increase only as an explicit behavior change.")
    parser.add_argument("--ssc_kappa_refresh_every", type=int, default=1,
                        help="κ-adaptive SSC: refresh per-pair cached c every N steps "
                             "(amortizes the eigvalsh+bisection across N steps). N=1 reproduces "
                             "per-step solving (default). Independent caches per Picard inner iter. "
                             "Only consulted when --polar_method ssc and --ssc_kappa are set.")
    parser.add_argument("--ssc_kappa_solver", type=str, default="eigvalsh",
                        choices=["eigvalsh", "misr_bisect", "stable_rank"],
                        help="κ-adaptive c solver. 'eigvalsh' = exact bisection on full "
                             "r×r spectrum (production default; launch-bound on small r). "
                             "'misr_bisect' = warm-started K-candidate bisection on MISR "
                             "F-norm (no eigvalsh). Best when launch-bound at small r. "
                             "'stable_rank' = stateless one-spike-plus-flat-tail c from "
                             "normalized stable rank; no eigvalsh, bisection, or c cache.")
    parser.add_argument("--ssc_kappa_bisect_iters", type=int, default=3,
                        help="K for --ssc_kappa_solver misr_bisect. Warm-started bisection in "
                             "log-c using K MISR runs. K=3 gives ~6%% accuracy with window=0.5.")
    parser.add_argument("--ssc_kappa_bisect_mode", type=str, default="sequential",
                        choices=["sequential", "parallel"],
                        help="Bisection strategy for --ssc_kappa_solver misr_bisect. "
                             "'sequential' = K MISR launches, classical bisection (default). "
                             "'parallel' = K log-spaced candidates per pair in one batched "
                             "MISR launch on (K*N, r, d); argmin |κ-target| picks the winner. "
                             "Parallel is coarser per K (residual log_window/(K-1) vs "
                             "log_window/2^K) but launches once — wins when launch-bound at "
                             "small r. To match seq-K=3 accuracy use par-K=9.")
    parser.add_argument("--ssc_kappa_bisect_nsteps_eval", type=int, default=None,
                        help="MISR nsteps for the parallel MISR-bisect κ-evaluator pass. "
                             "Default None means 2 × --ssc_nsteps. The winner apply pass "
                             "stays at --ssc_nsteps, preserving the production SSC apply "
                             "behavior while making candidate scoring less sensitive to "
                             "MISR under-convergence.")
    parser.add_argument("--ssc_kappa_cache_share_picard",
                        type=lambda s: str(s).lower() not in {"0", "false", "no"},
                        default=False,
                        help="κ-adaptive SSC: share the per-pair cached c across Picard inner "
                             "iterations (n=0 solves; n=1 reuses). The original snapshot test "
                             "measured DOWNSTREAM dA error (p50<1.1%%, p99<3.3%%) and concluded "
                             "share=True was safe. The in-training eigvalsh-comparison diagnostic "
                             "(--ssc_kappa_diagnose_eigvalsh) later showed share=True produces "
                             "factor 3-5x c-error on side A at n=1: side A's X spectrum changes "
                             "significantly between Picard iters due to the B^T @ dB @ A cross-"
                             "coupling correction, so reusing n=0's c at n=1 is meaningfully wrong. "
                             "Default flipped to False (correctness > 1.3%% wall savings); set "
                             "True only to reproduce legacy runs. Only consulted when "
                             "--polar_method ssc, --ssc_kappa set, and --picard_iters > 1.")
    parser.add_argument("--ssc_kappa_cache_ema_beta", type=float, default=None,
                        help="κ-adaptive SSC: when set, cache a log-space EMA of freshly "
                             "solved c values for warm-starts and non-refresh reuse steps. "
                             "Refresh steps still apply the freshly solved c; the EMA only "
                             "changes the cached value used between refreshes and as the "
                             "next kpar warm start. Default None preserves last-value cache "
                             "behavior.")
    parser.add_argument("--ssc_kappa_cross_group_eigvalsh",
                        type=lambda s: str(s).lower() not in {"0", "false", "no"},
                        default=True,
                        help="κ-adaptive SSC: batch eigvalsh across shape groups. "
                             "Stacks per-group (Ng, r, r) Grams into one (N_total, r, r) "
                             "eigvalsh + bisect per (side, picard iter) — at OLMo-2-1B "
                             "all-linear (3 shape groups, picard=2), collapses 12 small "
                             "eigvalsh launches into 4. Pure speedup at uniform r across "
                             "groups; silently disabled if r differs or fw_linearization=full. "
                             "Default True. Only consulted when --magnitude_rule "
                             "spectral_chord_tight_clean, --polar_method ssc, --ssc_kappa set, "
                             "and --ssc_kappa_solver eigvalsh.")
    parser.add_argument("--ssc_kappa_diagnose_eigvalsh",
                        type=lambda s: str(s).lower() not in {"0", "false", "no"},
                        default=False,
                        help="DIAGNOSTIC: at every polar call, also solve c via "
                             "eigvalsh on the same X and log per-pair |log(c_used) - "
                             "log(c_eigvalsh)|. Doubles eigvalsh work — only enable "
                             "for accuracy validation runs (e.g. kpar K=3 R=5). Per-step "
                             "events emit as JSONL `ssc_c_diag` with p50/p99/max log-error.")
    parser.add_argument("--ssc_kappa_diagnose_start_step", type=int, default=1,
                        help="When --ssc_kappa_diagnose_eigvalsh is enabled, "
                             "start emitting eigvalsh c-reference diagnostics "
                             "at this optimizer step. Default 1 preserves "
                             "existing behavior.")
    parser.add_argument("--ssc_kappa_diag_ema_beta", type=float, default=None,
                        help="DIAGNOSTIC: when --ssc_kappa_diagnose_eigvalsh is enabled, "
                             "also maintain a per-cache log-space EMA of eigvalsh true c "
                             "and log errors against that smoothed reference. Default None "
                             "logs only instantaneous true-c errors.")
    parser.add_argument("--ssc_kappa_warmup_steps", type=int, default=5,
                        help="κ-adaptive SSC: refresh every step for the first M steps before "
                             "honoring --ssc_kappa_refresh_every. At LoRA init the polar input's "
                             "spectrum is rank-deficient ⇒ κ-target unreachable ⇒ bisection "
                             "saturates at c_lo and a degenerate c gets cached. Default 5 covers "
                             "the early-step spread-out window. Only consulted when --ssc_kappa "
                             "and --ssc_kappa_refresh_every > 1.")
    parser.add_argument("--polar_sigma_power", type=float, default=None,
                        help="HTMuon (arXiv:2603.10067) σ → σ^p generalized polar. "
                             "None = use Newton-Schulz (default Muon polar). "
                             "0 = exact polar via SVD. p ∈ (0,1) = heavier-tailed update. "
                             "1 = no orthogonalization. HTMuon paper default p=0.125.")
    parser.add_argument("--polar_norm_dir", type=str, default="frob",
                        choices=["frob", "row", "col", "row_col", "col_row"],
                        help="Muon+ (arXiv:2602.21545) post-orthogonalization normalization "
                             "direction applied in adam-polar-product-lora's _polar_pipeline. "
                             "'frob' = original Frobenius RMS-align (default, no change). "
                             "'row'/'col' = unit-ℓ₂ per row/col of the orthogonalized output, "
                             "then rescale to ‖u‖_F. 'row_col'/'col_row' = composed.")
    parser.add_argument("--wandb_project", default=None, help="W&B project name. Omit to disable W&B.")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name. Auto-generated from key params if omitted.")
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Directory for step-continuous checkpoints (one `ckpt_step{N}` "
             "subdir per save). If unset, no checkpoints written. Use with "
             "--resume_from to recover from SLURM wall-timeouts. Normal "
             "resume reseeds via (seed, step); pass --resume_debug_replay "
             "for opt-in bitwise-debug replay.",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=None,
        help="Save cadence in steps. Default = --eval_every (one save per "
             "eval). Set higher (e.g. 1000) to save less often than eval — "
             "useful for long runs where save I/O is non-trivial. The final "
             "step always saves regardless of this cadence.",
    )
    parser.add_argument(
        "--checkpoint_keep_last", type=int, default=2,
        help="Keep this many most-recent checkpoints; delete older. Default 2 "
             "= one as a stable fallback, one in-flight. Set 0 to keep all.",
    )
    parser.add_argument(
        "--keep_checkpoints", action="store_true",
        help="By default, the checkpoint dir is deleted after the run reaches "
             "--max_steps (or --target_eval_loss). Pass this flag to retain "
             "checkpoints for later inspection or warm-start use. Failed/"
             "aborted runs (NaN abort, etc.) always retain checkpoints "
             "regardless of this flag.",
    )
    parser.add_argument(
        "--resume_from", default=None,
        help="Path to a specific checkpoint dir, OR a parent dir containing "
             "`ckpt_step{N}` children (auto-picks the latest). Idempotent: if "
             "the path doesn't exist or has no checkpoints, runs from step 1. "
             "Lets the same launch command work on first submission and on "
             "resubmission after wall-timeout.",
    )
    parser.add_argument(
        "--resume_replay_original_dataloader", action="store_true",
        help="DEBUG: when resuming, preserve the original single-pass dataloader "
             "order by keeping the base seed/sampler epoch and skipping "
             "resumed_step * grad_accum_steps microbatches before training. "
             "This is intended for deterministic failure localization, not "
             "normal wall-timeout resume.",
    )
    parser.add_argument(
        "--resume_debug_replay", action="store_true",
        help="DEBUG: bitwise-oriented resume path. Restores checkpointed "
             "Python/NumPy/torch RNG state after rebuilding and aligning the "
             "original dataloader stream. Implies "
             "--resume_replay_original_dataloader. Old checkpoints without "
             "RNG state still load but cannot fully restore RNG.",
    )
    parser.add_argument(
        "--snapshot_steps", default="",
        help="Comma-separated step indices at which to write a diagnostic "
             "snapshot (pre-step A, B and pre-σmax u_A, u_B per pair, plus "
             "the standard pair_state/group_state). Snapshots are written to "
             "<snapshot_dir>/step_{N} and are never pruned. Empty = disabled.",
    )
    parser.add_argument(
        "--snapshot_dir", default=None,
        help="Directory for diagnostic snapshots. Required when "
             "--snapshot_steps is non-empty.",
    )
    return parser


def main():
    args = make_parser().parse_args()
    resume_replay_original_dataloader = _resume_replays_original_dataloader(args)
    resume_restore_rng_state = _resume_restores_rng_state(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    # Initialize the distributed process group if launched under torchrun /
    # SLURM with WORLD_SIZE > 1. No-op otherwise (single-GPU back-compat).
    rank, world_size, local_rank = init_distributed()

    # If --global_batch_size is set, derive grad_accum_steps so that
    # batch_size × grad_accum × world_size == global_batch_size. The per-rank
    # batch_size keeps its CLI value (sized to fit GPU memory); grad_accum is
    # the lever we adjust to hit the target global batch.
    if args.global_batch_size is not None:
        denom = args.batch_size * world_size
        if args.global_batch_size % denom != 0:
            raise ValueError(
                f"--global_batch_size {args.global_batch_size} is not divisible "
                f"by per-rank batch_size {args.batch_size} × world_size "
                f"{world_size} = {denom}. Adjust --batch_size or "
                f"--global_batch_size so they divide cleanly."
            )
        derived_accum = args.global_batch_size // denom
        if derived_accum < 1:
            raise ValueError(
                f"--global_batch_size {args.global_batch_size} is smaller than "
                f"per-rank batch_size {args.batch_size} × world_size "
                f"{world_size} = {denom}; cannot derive a positive "
                f"grad_accum_steps."
            )
        args.grad_accum_steps = derived_accum

    if args.debug_higham_residual:
        from lora_playground import utils as _utils
        _utils.HIGHAM_DEBUG["enabled"] = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    # Under DDP, pin to the LOCAL_RANK device. init_distributed() already
    # called torch.cuda.set_device(local_rank); make `device` consistent.
    if device.type == "cuda" and world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    if device.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_bf16 = args.bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_dir:
        train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
        eval_dataset = load_from_disk(os.path.join(args.data_dir, "eval"))
    else:
        train_raw, eval_raw = load_splits(args)
        if args.data_pipeline_version.startswith("packed_v1"):
            train_dataset, eval_dataset = tokenize_splits_with_boundary(
                train_raw, eval_raw, tokenizer, args,
            )
        else:
            train_dataset, eval_dataset = tokenize_splits(
                train_raw, eval_raw, tokenizer, args,
            )

    # Under packed_v1+, the train dataset is packed-slot rows: each row
    # has `input_ids`, `labels`, `position_ids`, `doc_lens` all at length
    # seq_length (except doc_lens which is variable). Two routes:
    #
    #   (a) packed_v1.1+ (preferred): the on-disk cache is ALREADY packed
    #       at prepare_data time. train.py skips re-packing.
    #   (b) packed_v1 legacy: cache holds per-doc rows; pack at startup.
    #       Wastes CPU per run + non-deterministic slot ordering across
    #       seeds. Kept for backward compat loading of pre-2026-05-14
    #       caches.
    #
    # Detection: a pre-packed cache has the `labels` column; an unpacked
    # cache has only `input_ids` + `prompt_len`.
    docs_per_slot_mean = None
    is_packed_pipeline = args.data_pipeline_version.startswith("packed_v1")
    if is_packed_pipeline:
        cache_is_prepacked = (
            "labels" in train_dataset.column_names
            and "position_ids" in train_dataset.column_names
        )
        if cache_is_prepacked:
            n_slots = len(train_dataset)
            if is_main():
                print(
                    f"# {args.data_pipeline_version}: cache pre-packed at "
                    f"prepare_data time ({n_slots} slots @ seq={args.max_seq_length})",
                    flush=True,
                )
        else:
            n_docs_pre = len(train_dataset)
            train_dataset = pack_train_dataset(
                train_dataset,
                seq_length=args.max_seq_length,
                pad_token_id=tokenizer.pad_token_id,
            )
            n_slots = len(train_dataset)
            docs_per_slot_mean = (n_docs_pre / max(n_slots, 1)) if n_slots else None
            if is_main():
                print(
                    f"# {args.data_pipeline_version}: {n_docs_pre} docs → "
                    f"{n_slots} slots at train-time pack (mean "
                    f"{docs_per_slot_mean:.2f} docs/slot @ seq={args.max_seq_length}). "
                    f"Prefer a packed cache (packed_v1.1+) to avoid this re-pack.",
                    flush=True,
                )

    # Single-pass invariant: all training in this project must be one epoch
    # or less. Multi-epoch sweeps mix capacity-to-fit-the-subset effects with
    # per-step optimization quality, which makes optimizer comparisons
    # uninterpretable. (See: 8k-step runs invalidated 2026-05-04 because they
    # ran ~3.5 epochs over the 32k-sample subset.) Refuse to start if the
    # global step counter would consume more dataset units than exist.
    # Override with --allow_multi_epoch only for explicit experiments.
    #
    # Under packed_v1, len(train_dataset) is the number of PACKED SLOTS, not
    # docs. Each doc lives in exactly one slot, so single-pass-at-slot-level
    # is single-pass-at-doc-level — the math is the same, only the unit
    # changes. The error wording adapts so the reported number matches the
    # underlying counter.
    units_consumed = (
        args.max_steps * args.batch_size * args.grad_accum_steps * world_size
    )
    unit_name = "slot" if args.data_pipeline_version.startswith("packed_v1") else "sample"
    if units_consumed > len(train_dataset) and not getattr(args, "allow_multi_epoch", False):
        n_units = len(train_dataset)
        msg = (
            f"Multi-epoch training blocked: max_steps × batch_size × grad_accum × world_size "
            f"= {args.max_steps} × {args.batch_size} × {args.grad_accum_steps} × {world_size} = "
            f"{units_consumed:,} {unit_name}s, but train dataset has only "
            f"{n_units:,} {unit_name}s "
            f"(~{units_consumed / max(n_units, 1):.2f} epochs). "
        )
        if args.data_pipeline_version.startswith("packed_v1") and docs_per_slot_mean is not None:
            msg += (
                f"Note: under packed_v1, {n_units:,} slots ≈ "
                f"{int(n_units * docs_per_slot_mean):,} docs (mean "
                f"{docs_per_slot_mean:.2f} docs/slot). "
            )
        msg += (
            "Either reduce --max_steps, increase --max_train_samples / dataset "
            "size, or pass --allow_multi_epoch if multi-epoch is the intent."
        )
        raise ValueError(msg)

    # Collators: packed_v1 uses PackingCollator (train, pre-packed slots)
    # + PadToMaxCollator (eval, per-doc pad-to-max for static shape under
    # compile). unpacked_v0 keeps the legacy dynamic-shape path.
    if args.data_pipeline_version.startswith("packed_v1"):
        # Mask dtype tracks the model dtype so additive 4D mask casts
        # cleanly inside SDPA. bf16 model → bf16 mask; fp32 model → fp32.
        mask_dtype = torch.bfloat16 if use_bf16 else torch.float32
        train_collator = PackingCollator(
            seq_length=args.max_seq_length,
            mask_dtype=mask_dtype,
        )
        eval_collator = PadToMaxCollator(
            seq_length=args.max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        legacy_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        train_collator = legacy_collator
        eval_collator = legacy_collator
    # Under DDP each rank reads a disjoint shard of the dataset via
    # DistributedSampler. Single-process keeps the original simple shuffle.
    # Workers per process are scaled down so we don't oversubscribe CPUs.
    per_proc_workers = max(1, args.num_workers // max(world_size, 1))
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
    else:
        train_sampler = None
        eval_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=train_collator,
        num_workers=per_proc_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=eval_collator,
        num_workers=per_proc_workers,
        pin_memory=device.type == "cuda",
    )

    parsed_target_modules = parse_target_modules(args.target_modules)
    svd_rank = args.svd_rank if args.svd_rank is not None else args.lora_r

    peft = build_peft_model(
        model_name=args.model_name,
        training_mode=args.training_mode,
        target_modules=parsed_target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        use_liger=args.use_liger,
        liger_flce=False,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_mode=(args.compile_mode if args.compile else None),
        device=device,
        world_size=world_size,
        local_rank=local_rank,
    )
    if args.compile:
        from lora_playground.optim import enable_kernel_compile
        enable_kernel_compile()
    bare_model = peft.bare_model
    model = peft.train_model
    liger_applied = peft.liger_applied
    dense_targets = peft.dense_targets
    ucv_target_names = peft.ucv_target_names

    if args.training_mode == "lora" and args.lora_init_b != "zero":
        from lora_playground.utils import override_lora_init
        overrides = override_lora_init(bare_model, args.lora_init_b)
        log_event({
            "event": "lora_init_override",
            "mode": args.lora_init_b,
            "n_layers": len(overrides),
            "delta_w0_frob_total": sum(x[1] for x in overrides),
        })

    if args.training_mode == "lora":
        if args.optimizer in {"svd-step-adamw", "svd-cumulative-adamw", "galore-adamw", "adam-ucv-core-lora"}:
            raise ValueError(f"{args.optimizer} requires a non-lora training_mode.")
        effective_optimizer = args.optimizer
        rank_constraint = None
    elif args.training_mode == "ucv":
        if args.optimizer != "adam-ucv-core-lora":
            raise ValueError("ucv training_mode requires --optimizer adam-ucv-core-lora.")
        effective_optimizer = "adam-ucv-core-lora"
        rank_constraint = "ucv_orthogonal_core"
    elif args.training_mode == "svd_step_oracle":
        if args.optimizer not in {"adamw", "svd-step-adamw"}:
            raise ValueError("svd_step_oracle currently supports AdamW only.")
        effective_optimizer = "svd-step-adamw"
        rank_constraint = "per_step_update"
    elif args.training_mode == "svd_cumulative_oracle":
        if args.optimizer not in {"adamw", "svd-cumulative-adamw"}:
            raise ValueError("svd_cumulative_oracle currently supports AdamW only.")
        effective_optimizer = "svd-cumulative-adamw"
        rank_constraint = "cumulative_displacement"
    else:  # galore
        if args.optimizer not in {"adamw", "galore-adamw"}:
            raise ValueError("galore training_mode requires --optimizer galore-adamw.")
        effective_optimizer = "galore-adamw"
        rank_constraint = "galore_projection"

    svd_niter = None if args.svd_niter < 0 else args.svd_niter
    # Pass the bare PEFT model (not DDP/compile-wrapped) so the optimizer's
    # lora_A/lora_B traversal works regardless of wrapping. The pair_state
    # tensors hold direct references to the underlying nn.Parameter objects,
    # so DDP's gradient hooks still update them in place.
    optimizer = build_optimizer(
        bare_model,
        optimizer_type=effective_optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scaled_metric=args.scaled_metric,
        lora_plus_multiplier=args.lora_plus_multiplier,
        targets=dense_targets if dense_targets else None,
        svd_rank=svd_rank if dense_targets else None,
        svd_niter=svd_niter if dense_targets else 4,
        precond_gamma=args.precond_gamma,
        precond_ema_beta=args.precond_ema_beta,
        precond_delta=args.precond_delta,
        psi_inner_iters=args.psi_inner_iters,
        psi_momentum=args.psi_momentum,
        psi_rho=args.psi_rho,
        psi_momentum_rank=args.psi_momentum_rank,
        galore_update_proj_gap=args.galore_update_proj_gap,
        galore_scale=args.galore_scale,
        muon_ns_steps=args.muon_ns_steps,
        ns_form=args.ns_form,
        muon_alpha=args.lora_alpha,
        muon_rank=args.lora_r,
        log_basic_diagnostics=args.log_basic_diagnostics,
        log_heavy_diagnostics=args.log_heavy_diagnostics,
        optim_diagnostics_every=args.optim_diagnostics_every,
        precond_refresh_every=args.precond_refresh_every,
        precond_method=args.precond_method,
        precond_delta_relative=args.precond_delta_relative,
        curvature_whitening=args.curvature_whitening,
        curvature_beta=args.curvature_beta,
        higham_compute_dtype=args.higham_compute_dtype,
        log_non_finite=args.log_non_finite,
        log_non_finite_start_step=args.log_non_finite_start_step,
        debug_optimizer_state=args.debug_optimizer_state,
        debug_optimizer_state_every=args.debug_optimizer_state_every,
        debug_optimizer_state_start_step=args.debug_optimizer_state_start_step,
        debug_snapshot_dir=args.debug_snapshot_dir,
        debug_snapshot_limit=args.debug_snapshot_limit,
        debug_abort_on_non_finite=args.debug_abort_on_non_finite,
        higham_iters=args.higham_iters,
        picard_alpha=args.picard_alpha,
        htmuon_p=args.htmuon_p,
        picard_iters_override=args.picard_iters_override,
        cw_picard_iters=args.cw_picard_iters,
        cw_nesterov=args.cw_nesterov,
        cw_no_radius=args.cw_no_radius,
        cw_no_diag_curv=args.cw_no_diag_curv,
        precond=args.precond,
        msign=args.msign,
        cw_unpinned=args.cw_unpinned,
        cw_solved_rho=args.cw_solved_rho,
        cw_factor_a=args.cw_factor_a,
        cw_factor_b=args.cw_factor_b,
        rdinv_variant=args.rdinv_variant,
        rdinv_delta=args.rdinv_delta,
        cw_metric_init=args.cw_metric_init,
        dump_pre_polar_dir=args.dump_pre_polar_dir,
        dump_pre_polar_every=args.dump_pre_polar_every,
        dump_pre_polar_pairs=args.dump_pre_polar_pairs,
        dump_pre_polar_max_pairs=args.dump_pre_polar_max_pairs,
        anderson_m=args.anderson_m,
        anderson_reg=args.anderson_reg,
        soap_beta=args.soap_beta,
        soap_refresh_every=args.soap_refresh_every,
        polar_norm_dir=args.polar_norm_dir,
        polar_sigma_power=args.polar_sigma_power,
        polar_method=args.polar_method,
        polar_core_remix_alpha=args.polar_core_remix_alpha,
        ssc_c=args.ssc_c,
        ssc_nsteps=args.ssc_nsteps,
        ssc_kappa=args.ssc_kappa,
        ssc_kappa_refresh_every=args.ssc_kappa_refresh_every,
        ssc_kappa_warmup_steps=args.ssc_kappa_warmup_steps,
        ssc_kappa_solver=args.ssc_kappa_solver,
        ssc_kappa_bisect_iters=args.ssc_kappa_bisect_iters,
        ssc_kappa_bisect_mode=args.ssc_kappa_bisect_mode,
        ssc_kappa_bisect_nsteps_eval=args.ssc_kappa_bisect_nsteps_eval,
        ssc_kappa_cache_share_picard=args.ssc_kappa_cache_share_picard,
        ssc_kappa_cache_ema_beta=args.ssc_kappa_cache_ema_beta,
        ssc_kappa_cross_group_eigvalsh=args.ssc_kappa_cross_group_eigvalsh,
        ssc_kappa_diagnose_eigvalsh=args.ssc_kappa_diagnose_eigvalsh,
        ssc_kappa_diagnose_start_step=args.ssc_kappa_diagnose_start_step,
        ssc_kappa_diag_ema_beta=args.ssc_kappa_diag_ema_beta,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    if args.optim_heldout_probe:
        if get_world_size() != 1:
            raise ValueError("--optim_heldout_probe currently requires one process")
        if args.optim_heldout_probe_batches < 1:
            raise ValueError("--optim_heldout_probe_batches must be at least 1")
        if not args.log_heavy_diagnostics or getattr(optimizer, "precond", None) != "factorwise":
            raise ValueError(
                "--optim_heldout_probe requires --log_heavy_diagnostics and "
                "--precond factorwise"
            )
        if (args.optim_heldout_identity_scale is not None
                and (not math.isfinite(args.optim_heldout_identity_scale)
                     or args.optim_heldout_identity_scale <= 0)):
            raise ValueError("--optim_heldout_identity_scale must be finite and positive")
    elif args.optim_heldout_probe_exit or args.optim_heldout_identity_scale is not None:
        raise ValueError(
            "--optim_heldout_probe_exit and --optim_heldout_identity_scale "
            "require --optim_heldout_probe"
        )
    if args.optim_small_slot_microbatch_probe:
        if not args.optim_heldout_probe:
            raise ValueError(
                "--optim_small_slot_microbatch_probe requires --optim_heldout_probe")
        if args.lora_r != 16:
            raise ValueError(
                "--optim_small_slot_microbatch_probe is the checkpoint-local r=16 probe")
        if args.grad_accum_steps < 2:
            raise ValueError(
                "--optim_small_slot_microbatch_probe requires grad_accum_steps >= 2 "
                "to distinguish centered and uncentered microbatch moments")
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    wandb_run = None
    # Wandb on rank 0 only — every rank logging the same run would either
    # corrupt the run state or require per-rank child runs that downstream
    # tooling doesn't expect.
    if args.wandb_project and is_main():
        import wandb

        run_name = args.wandb_run_name or (
            f"{args.optimizer}_lr{args.lr}_lp{args.lora_plus_multiplier}_s{args.seed}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            resume="never",
        )

    dirty_state = git_dirty_state()
    _attempt = _current_attempt_metadata(args)
    _semantic_revisions = semantic_revisions(
        optimizer,
        {"data_pipeline_version": args.data_pipeline_version},
    )
    _git_commit = git_commit() or "unavailable"
    _optimizer_provenance = _optimizer_provenance_fields(
        optimizer_name=effective_optimizer,
        optimizer=optimizer,
        semantic_revision=_semantic_revisions["optimizer_impl"],
        implementation_revision=_git_commit,
    )
    log_event(
        {
            "event": "config",
            **_attempt,
            "semantic_revisions": _semantic_revisions,
            # Scalar projections keep semantic revisions visible to existing
            # dedup/series identity code, which deliberately ignores nested
            # config blocks. The bounded dict above remains the lineage API.
            "optimizer_impl_revision": _semantic_revisions["optimizer_impl"],
            "measurement_semantics_revision": _semantic_revisions["measurement"],
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "git_commit": _git_commit,
            # Recorded provenance, never an admission or attestation gate.
            "git_dirty": dirty_state["git_dirty"],
            "git_status": dirty_state["git_status"],
            "device": str(device),
            "training_mode": args.training_mode,
            "optimizer": effective_optimizer,
            "requested_optimizer": args.optimizer,
            **_optimizer_provenance,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name if not args.train_file else "json",
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "data_dir": args.data_dir,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "svd_rank": svd_rank if dense_targets else None,
            "rank_constraint": rank_constraint,
            "target_module_count": len(dense_targets) if dense_targets else len(ucv_target_names),
            "target_module_names": (
                [target.name for target in dense_targets] if dense_targets else ucv_target_names
            ),
            "svd_projection": ("exact" if svd_niter is None else f"randomized_niter{svd_niter}") if dense_targets else None,
            "exclude_lm_head_from_all_linear": (
                bool(dense_targets and parsed_target_modules == "all-linear")
                or bool(ucv_target_names and parsed_target_modules == "all-linear")
            ),
            "lr": args.lr,
            "lora_plus_multiplier": args.lora_plus_multiplier,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "bf16": use_bf16,
            "tf32": device.type == "cuda" and not args.no_tf32,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "attn_implementation": getattr(bare_model.config, "_attn_implementation",
                                           args.attn_implementation),
            "liger_family": liger_applied,
            "world_size": world_size,
            "global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
            "per_rank_batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "data_pipeline_version": args.data_pipeline_version,
            "docs_per_slot_mean": docs_per_slot_mean,
            "profile_steps": args.profile_steps,
            "picard_alpha": args.picard_alpha,
            "picard_iters_override": args.picard_iters_override,
            # EFFECTIVE values, read off the BUILT optimizer (not args.*): some
            # branches drop a flag (a hardcoded literal or omitted kwarg), so logging
            # the CLI arg records a value the optimizer never used — the beta1 lie
            # class. getattr(optimizer, attr, args.*) makes the config event
            # self-truthing; the args fallback covers optimizers without the attr.
            "cw_picard_iters": getattr(optimizer, "cw_picard_iters", args.cw_picard_iters),
            "cw_nesterov": getattr(optimizer, "cw_nesterov", args.cw_nesterov),
            "cw_no_radius": getattr(optimizer, "cw_no_radius", args.cw_no_radius),
            "cw_no_diag_curv": getattr(optimizer, "cw_no_diag_curv", args.cw_no_diag_curv),
            # Record the RESOLVED branch, not the CLI value: `precond=None` means
            # "inherit the optimizer spec", so args.precond alone would log None
            # for every default run and lose which of the three actually ran.
            "precond": getattr(optimizer, "precond", args.precond),
            "msign": getattr(optimizer, "msign", args.msign),
            "cw_unpinned": getattr(optimizer, "cw_unpinned", args.cw_unpinned),
            "cw_solved_rho": getattr(optimizer, "cw_solved_rho", args.cw_solved_rho),
            "cw_factor_a": getattr(optimizer, "cw_factor_a", args.cw_factor_a),
            "cw_factor_b": getattr(optimizer, "cw_factor_b", args.cw_factor_b),
            "anderson_m": args.anderson_m,
            "anderson_reg": args.anderson_reg,
            "soap_beta": args.soap_beta,
            "soap_refresh_every": args.soap_refresh_every,
            "polar_norm_dir": args.polar_norm_dir,
            "polar_sigma_power": args.polar_sigma_power,
            "polar_method": args.polar_method,
            "ssc_c": args.ssc_c,
            "ssc_nsteps": args.ssc_nsteps,
            "ssc_kappa": args.ssc_kappa,
            "ssc_kappa_refresh_every": args.ssc_kappa_refresh_every,
            "ssc_kappa_warmup_steps": args.ssc_kappa_warmup_steps,
            "ssc_kappa_solver": args.ssc_kappa_solver,
            "ssc_kappa_bisect_iters": args.ssc_kappa_bisect_iters,
            "ssc_kappa_bisect_mode": args.ssc_kappa_bisect_mode,
            "ssc_kappa_bisect_nsteps_eval": args.ssc_kappa_bisect_nsteps_eval,
            "ssc_kappa_cache_share_picard": args.ssc_kappa_cache_share_picard,
            "ssc_kappa_cache_ema_beta": args.ssc_kappa_cache_ema_beta,
            "ssc_kappa_cross_group_eigvalsh": args.ssc_kappa_cross_group_eigvalsh,
            "ssc_kappa_diagnose_eigvalsh": args.ssc_kappa_diagnose_eigvalsh,
            "ssc_kappa_diagnose_start_step": args.ssc_kappa_diagnose_start_step,
            "ssc_kappa_diag_ema_beta": args.ssc_kappa_diag_ema_beta,
            "log_non_finite_start_step": args.log_non_finite_start_step,
            "debug_optimizer_state_start_step": args.debug_optimizer_state_start_step,
            "polar_core_remix_alpha": args.polar_core_remix_alpha,
            # EFFECTIVE (off the built optimizer) — see the cw_* note above. beta1
            # was hardcoded 0.9 in the curvature-whiten branches and 0.95 in
            # muon-coupled-core while args said otherwise; precond_delta is hardcoded
            # 1e-6 in the lin/scaled/coupled-core families; curvature_whitening is
            # dropped by every AdamPolarProductLoRA branch except chord-tight-clean.
            "beta1": getattr(optimizer, "beta1", args.beta1),
            "beta2": getattr(optimizer, "beta2", args.beta2),
            "lora_init_b": args.lora_init_b,
            "precond_delta": getattr(optimizer, "delta", args.precond_delta),
            "precond_delta_relative": getattr(optimizer, "precond_delta_relative", args.precond_delta_relative),
            "curvature_whitening": getattr(optimizer, "curvature_whitening", args.curvature_whitening),
            "curvature_beta": args.curvature_beta,
            # Diagnostics flags under canonical names. The loader's read-side
            # alias chain (log_optim_diagnostics → log_basic_diagnostics,
            # log_diagnostics → log_basic_diagnostics, diagnostics_every →
            # optim_diagnostics_every) exists only for legacy cfgs; new
            # runs always emit these canonical keys.
            "diagnostics": {
                "basic": bool(args.log_basic_diagnostics),
                "heavy": bool(args.log_heavy_diagnostics),
                "every": int(args.optim_diagnostics_every),
                "heldout_probe": bool(args.optim_heldout_probe),
                "heldout_probe_batches": int(args.optim_heldout_probe_batches),
                "heldout_probe_exit": bool(args.optim_heldout_probe_exit),
                "heldout_identity_scale": args.optim_heldout_identity_scale,
                "small_slot_microbatch_probe": bool(
                    args.optim_small_slot_microbatch_probe),
            },
            # Future-proofing: blanket dump of every CLI flag so analysis
            # never has to wait for a manual cfg event update when a new
            # flag is added. Named fields above remain for backward compat.
            "_cli_args": {k: v for k, v in vars(args).items()
                          if not k.startswith("_") and not callable(v)},
        }
    )

    # MFU numerator: model FLOPs per step ≈ c · N_total · tokens_per_step,
    # where c is 4 for LoRA/UCV (frozen base, no param-grad pass) or 6
    # for full FT, +2 if gradient checkpointing is on. We capture N_total
    # once here so the eval-loop math is just a divide. Counts the bare
    # model (frozen base + LoRA), not the DDP/compile wrappers.
    mfu_n_params = count_total_params(bare_model)
    mfu_peak_tflops = device_peak_tflops() if device.type == "cuda" else None
    mfu_flops_per_token_per_param = flops_per_token_for_mode(
        args.training_mode, args.gradient_checkpointing,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    profiler = None
    if args.profile_steps > 0:
        from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

        os.makedirs(args.profile_dir, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profiler = profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()

    # Resume from checkpoint if requested. Idempotent: missing path / empty
    # dir returns None, so the same launch command works on first submission
    # and on resubmission after wall-timeout. Resume MUST happen after
    # optimizer / scheduler construction (they're the load targets) but before
    # any RNG / sampler initialization that depends on (seed, step).
    resume_state = None
    resume_segment = 0
    if args.resume_from is not None:
        resume_state = load_checkpoint(
            args.resume_from,
            bare_model=bare_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if resume_state is not None:
            loaded_checkpoint_identity = resume_state.get("checkpoint_identity")
            if (loaded_checkpoint_identity is not None
                    and loaded_checkpoint_identity != _attempt["checkpoint_identity"]):
                raise ValueError(
                    "checkpoint lineage mismatch: current attempt declares "
                    f"{_attempt['checkpoint_identity']!r}, but loaded checkpoint "
                    f"declares {loaded_checkpoint_identity!r}"
                )
            resume_parent_attempt_id = resume_state.get("attempt_id")
            if resume_parent_attempt_id == _attempt["attempt_id"]:
                raise ValueError(
                    "checkpoint attempt metadata would create a self-parent "
                    f"lineage for {_attempt['attempt_id']!r}"
                )
            resume_segment = resume_state["resume_segment"] + 1
            log_event({
                "event": "resume",
                "attempt_id": _attempt["attempt_id"],
                "resume_parent_attempt_id": resume_parent_attempt_id,
                "checkpoint_identity": _attempt["checkpoint_identity"],
                "checkpoint_metadata_explicit": bool(
                    resume_parent_attempt_id is not None
                    and loaded_checkpoint_identity is not None
                ),
                "resumed_from_step": resume_state["step"],
                "resumed_from_total_tokens": resume_state["total_tokens"],
                "resume_segment": resume_segment,
                "ckpt_path": resume_state["ckpt_path"],
            })
            if resume_restore_rng_state:
                log_event({
                    "event": "resume_debug_replay",
                    "resumed_from_step": resume_state["step"],
                    "rng_state_present": bool(
                        resume_state.get("rng_state_present", False)
                    ),
                    "replay_original_dataloader": True,
                })
            if resume_replay_original_dataloader:
                log_event({
                    "event": "resume_replay_original_dataloader",
                    "resumed_from_step": resume_state["step"],
                    "microbatches_to_skip": (
                        resume_state["step"] * args.grad_accum_steps
                    ),
                })
            else:
                # Reseed so the post-resume sampler advances to a new
                # deterministic position. Continuous in expectation, not bitwise.
                set_seed(args.seed + resume_state["step"])
    start_step = (resume_state["step"] + 1) if resume_state else 1

    # Under DDP, set_epoch on the DistributedSampler ensures different shuffle
    # ordering across epochs (only matters if we ever multi-epoch; the project
    # invariant is 1-pass so this is just a forward-compat call).
    sampler_epoch = (
        0
        if resume_state is not None and resume_replay_original_dataloader
        else resume_segment
    )
    if train_sampler is not None:
        train_sampler.set_epoch(sampler_epoch)
    train_iter = iter(train_loader)
    heldout_probe_iter = iter(eval_loader) if args.optim_heldout_probe else None
    if resume_state is not None and resume_replay_original_dataloader:
        microbatches_to_skip = resume_state["step"] * args.grad_accum_steps
        skipped = 0
        for _ in range(microbatches_to_skip):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)
            skipped += 1
        log_event({
            "event": "resume_replay_dataloader_aligned",
            "resumed_from_step": resume_state["step"],
            "sampler_epoch": sampler_epoch,
            "microbatches_skipped": skipped,
        })
    if resume_state is not None and resume_restore_rng_state:
        if resume_state.get("rng_state") is not None:
            rng_status = restore_rng_state(resume_state["rng_state"])
            log_event({
                "event": "resume_debug_rng_restored",
                "resumed_from_step": resume_state["step"],
                **rng_status,
            })
        else:
            log_event({
                "event": "resume_debug_rng_missing",
                "resumed_from_step": resume_state["step"],
            })
    total_tokens = resume_state["total_tokens"] if resume_state else 0
    eval_elapsed = 0.0
    # Windowed train-loss accumulator for `train_step` events.
    window_loss_sum = 0.0
    window_tokens = 0
    cuda_sync()
    start = time.perf_counter()
    model.train()

    if start_step > args.max_steps:
        log_event({
            "event": "resume_already_complete",
            "step": resume_state["step"],
            "max_steps": args.max_steps,
        })

    # Checkpoint cadence: default to --eval_every (one save per eval). User
    # can override to e.g. 1000 to save less often than eval.
    ckpt_every = args.checkpoint_every or args.eval_every

    # Diagnostic snapshot driver. The optimizer carries a flag
    # `snapshot_pair_tensors` that, when True for one step, stashes pre-step
    # A, B and pre-σmax u_A, u_B into pair_state. We flip it on right before
    # the step and write the snapshot dir right after, then clear the stash.
    # Step 0 (if requested) is handled in-loop by triggering on step 1 and
    # naming the snapshot dir step_0 — pre-step A, B at step 1 IS the init
    # state, and u_A_pre at step 1 is the first Adam direction.
    snapshot_step_set: set[int] = set()
    snapshot_step0_requested = False
    if args.snapshot_steps:
        toks = [t.strip() for t in args.snapshot_steps.split(",") if t.strip()]
        snapshot_step_set = {int(t) for t in toks}
        snapshot_step0_requested = 0 in snapshot_step_set
        snapshot_step_set.discard(0)
        if args.snapshot_dir is None:
            raise ValueError(
                "--snapshot_steps requires --snapshot_dir"
            )
    optimizer_supports_snapshot = hasattr(optimizer, "snapshot_pair_tensors")
    if snapshot_step_set and not optimizer_supports_snapshot:
        log_event({
            "event": "snapshot_unsupported_optimizer",
            "optimizer": type(optimizer).__name__,
        })
        snapshot_step_set = set()
        snapshot_step0_requested = False

    # Tracked across the loop: an early-break via --abort_on_nan_eval flips
    # this so end-of-run cleanup is skipped (preserve checkpoints for
    # debugging). Natural max_steps completion AND target_eval_loss hits are
    # treated as "complete" — checkpoints get deleted by default.
    run_completed_cleanly = True

    for step in range(start_step, args.max_steps + 1):
        # Diagnostic snapshot: flip the optimizer's stash flag before the step
        # so it captures A_pre, B_pre, u_A_pre, u_B_pre into pair_state for
        # the in-flight step. snapshot_label names the output subdir:
        #   - step ∈ snapshot_step_set → label = step
        #   - step == 1 and step-0 was requested → label = 0 (init state +
        #     first Adam direction)
        snapshot_label = None
        if optimizer_supports_snapshot:
            if step in snapshot_step_set:
                optimizer.snapshot_pair_tensors = True
                snapshot_label = step
            elif step == 1 and snapshot_step0_requested:
                optimizer.snapshot_pair_tensors = True
                snapshot_label = 0

        pre_step_callback = None
        heldout_batch = None
        if args.optim_heldout_probe and step % args.optim_diagnostics_every == 0:
            heldout_batch = []
            for _ in range(args.optim_heldout_probe_batches):
                try:
                    heldout_batch.append(next(heldout_probe_iter))
                except StopIteration:
                    heldout_probe_iter = iter(eval_loader)
                    heldout_batch.append(next(heldout_probe_iter))

            def pre_step_callback(batch=heldout_batch):
                attach_heldout_factor_grads(model, optimizer, batch, device)

        step_loss, step_tokens, train_iter, norm_stats = run_one_train_step(
            model, optimizer, train_iter, train_loader,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            device=device,
            pre_step_callback=pre_step_callback,
            capture_factor_microbatch_grads=(
                args.optim_small_slot_microbatch_probe
                and heldout_batch is not None
            ),
        )
        if heldout_batch is not None:
            heldout_direction_losses = measure_heldout_factor_directions(
                model, optimizer, heldout_batch, device,
                identity_scale=args.optim_heldout_identity_scale,
            )
            log_event({
                "event": "cw_shadow_heldout_loss",
                "step": step,
                **heldout_direction_losses,
            })
            if args.optim_heldout_probe_exit:
                log_event({"event": "optim_heldout_probe_exit", "step": step})
                break
        scheduler.step()
        total_tokens += step_tokens
        window_loss_sum += step_loss
        window_tokens += step_tokens
        if profiler is not None:
            profiler.step()

        # Per-window train-loss event (cheap; no held-out eval). Emitted on
        # every multiple of train_loss_every — including eval steps, so the
        # window length is consistent (last train_loss_every steps) and
        # doesn't grow when an eval step falls within the window.
        if args.train_loss_every > 0 and step % args.train_loss_every == 0:
            log_event({
                "event": "train_step",
                "step": step,
                "train_loss": window_loss_sum / max(window_tokens, 1),
                "tokens": total_tokens,
                "lr": scheduler.get_last_lr()[0],
            })
            if norm_stats:
                log_event({
                    "event": "train_norms",
                    "step": step,
                    **norm_stats,
                })
            if wandb_run is not None:
                wandb_run.log(
                    {"train_loss": window_loss_sum / max(window_tokens, 1)},
                    step=step,
                )
            window_loss_sum = 0.0
            window_tokens = 0

        if step % args.eval_every == 0 or step == args.max_steps:
            cuda_sync()
            train_elapsed = time.perf_counter() - start - eval_elapsed
            eval_start = time.perf_counter()
            eval_loss = evaluate(model, eval_loader, device)
            cuda_sync()
            eval_sec = time.perf_counter() - eval_start
            eval_elapsed += eval_sec
            peak_memory_mb = None
            if device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            # MFU diagnostic: 6·N·T / (step_time × peak_TFLOPS). Computed
            # over the cumulative train window so transient stalls don't
            # spike a single number. Skipped on CPU (no peak FLOPs to
            # divide by) or when peak lookup failed (unknown GPU). On
            # contended GPUs the number is meaningless — only trust under
            # exclusive allocation.
            # Use post-resume segment counters so MFU reflects current
            # throughput, not "ratio of segment-elapsed to absolute-step".
            _seg_steps_for_mfu = step - start_step + 1
            _seg_tokens_for_mfu = total_tokens - (
                resume_state["total_tokens"] if resume_state else 0
            )
            mean_step_time = train_elapsed / max(_seg_steps_for_mfu, 1)
            mean_tokens_per_step = _seg_tokens_for_mfu / max(_seg_steps_for_mfu, 1)
            if device.type == "cuda" and mfu_peak_tflops is not None:
                mfu = compute_mfu(
                    n_params=mfu_n_params,
                    tokens_per_step=int(mean_tokens_per_step),
                    step_time_sec=mean_step_time,
                    peak_tflops=mfu_peak_tflops,
                    flops_per_token_per_param=mfu_flops_per_token_per_param,
                )
            else:
                mfu = None
            # `train_elapsed_sec` and `tokens_per_sec` reflect the
            # POST-RESUME segment only (timer starts at loop entry). For
            # cumulative wall, subscribe to per-resume segments via
            # `resume_segment` + the loader's stitching.
            segment_steps = step - start_step + 1
            segment_tokens = total_tokens - (
                resume_state["total_tokens"] if resume_state else 0
            )
            eval_payload = {
                "event": "eval",
                "step": step,
                "train_loss": step_loss / max(step_tokens, 1),
                "eval_loss": eval_loss,
                "tokens": total_tokens,
                "train_elapsed_sec": train_elapsed,
                "eval_sec": eval_sec,
                "tokens_per_sec": segment_tokens / max(train_elapsed, 1e-9),
                "resume_segment": resume_segment,
                "peak_memory_mb": peak_memory_mb,
                "lr": scheduler.get_last_lr()[0],
                "mfu": mfu,
                "mfu_peak_tflops": mfu_peak_tflops,
                "mfu_n_params": mfu_n_params,
                "mfu_flops_per_token_per_param": mfu_flops_per_token_per_param,
            }
            log_event(eval_payload)
            if wandb_run is not None:
                wandb_run.log(
                    {k: v for k, v in eval_payload.items() if k not in ("event",)},
                    step=step,
                )
            if args.abort_on_nan_eval and not math.isfinite(eval_loss):
                log_event({
                    "event": "abort_on_nan_eval",
                    "step": step,
                    "eval_loss": eval_loss,
                })
                run_completed_cleanly = False
                break
            if (
                args.abort_on_eval_loss_above is not None
                and math.isfinite(eval_loss)
                and eval_loss > args.abort_on_eval_loss_above
            ):
                log_event({
                    "event": "abort_on_eval_loss_above",
                    "step": step,
                    "eval_loss": eval_loss,
                    "threshold": args.abort_on_eval_loss_above,
                })
                run_completed_cleanly = False
                break
            if args.target_eval_loss is not None and eval_loss <= args.target_eval_loss:
                break

        # Diagnostic snapshot save. Fires AFTER the step (and after eval) so
        # the optimizer's pair_state holds the freshly-stashed A, B (§9
        # factor inputs) and u_A, u_B (Adam-RMS direction, pre-σmax). Save dir is <snapshot_dir>/step_{label} where
        # label = 0 for the special-case step-0 request, else the step index.
        # Independent of --checkpoint_keep_last pruning.
        if (
            snapshot_label is not None
            and args.snapshot_dir is not None
            and is_main()
        ):
            snap_path = Path(args.snapshot_dir) / f"step_{snapshot_label}"
            try:
                save_checkpoint(
                    snap_path,
                    bare_model=bare_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    total_tokens=total_tokens,
                    resume_segment=resume_segment,
                    cfg_snapshot={"command": " ".join(
                        shlex.quote(a) for a in sys.argv
                    )},
                    attempt_id=_attempt["attempt_id"],
                    checkpoint_identity=_attempt["checkpoint_identity"],
                )
                log_event({
                    "event": "snapshot_saved",
                    "step": step,
                    "label": snapshot_label,
                    "path": str(snap_path),
                })
            except Exception as e:
                log_event({
                    "event": "snapshot_save_failed",
                    "step": step,
                    "label": snapshot_label,
                    "error": f"{type(e).__name__}: {e}",
                })
            finally:
                for ps in optimizer.pair_state.values():
                    for k in ("A", "B", "u_A", "u_B"):
                        ps.pop(k, None)
                optimizer.snapshot_pair_tensors = False

        # Checkpoint save. Cadence is `--checkpoint_every` if set, else
        # `--eval_every` so the default matches the eval+save coupling.
        # The final step always saves regardless. Save AFTER the eval block
        # so target_eval_loss / NaN-abort breaks bypass the save; that
        # preserves the "abort retains checkpoints" semantic of `--keep_*`.
        if (
            args.checkpoint_dir is not None
            and is_main()
            and (step % ckpt_every == 0 or step == args.max_steps)
        ):
            try:
                save_checkpoint(
                    ckpt_dir_for_step(args.checkpoint_dir, step),
                    bare_model=bare_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    total_tokens=total_tokens,
                    resume_segment=resume_segment,
                    cfg_snapshot={"command": " ".join(
                        shlex.quote(a) for a in sys.argv
                    )},
                    attempt_id=_attempt["attempt_id"],
                    checkpoint_identity=_attempt["checkpoint_identity"],
                )
                if args.checkpoint_keep_last and args.checkpoint_keep_last > 0:
                    prune_checkpoints(
                        args.checkpoint_dir, args.checkpoint_keep_last
                    )
            except Exception as e:
                log_event({
                    "event": "checkpoint_save_failed",
                    "step": step,
                    "error": f"{type(e).__name__}: {e}",
                })

    if profiler is not None:
        profiler.stop()
    if wandb_run is not None:
        wandb_run.finish()

    # End-of-run checkpoint cleanup. Successful completion deletes the
    # checkpoint dir (it's no longer needed for resume) unless the user
    # opted in via --keep_checkpoints. NaN-abort / target_eval_loss runs
    # are treated as "not cleanly complete" — for the early-target case
    # this is conservative, but the checkpoints are still useful as a
    # warm-start so retaining them costs only disk.
    if (
        args.checkpoint_dir is not None
        and is_main()
        and run_completed_cleanly
        and not args.keep_checkpoints
    ):
        import shutil
        try:
            shutil.rmtree(args.checkpoint_dir)
            log_event({
                "event": "checkpoints_cleaned",
                "checkpoint_dir": args.checkpoint_dir,
            })
        except OSError as e:
            log_event({
                "event": "checkpoint_cleanup_failed",
                "checkpoint_dir": args.checkpoint_dir,
                "error": f"{type(e).__name__}: {e}",
            })

    # Tear down the distributed process group if we initialized one. No-op
    # in single-process mode.
    dist_cleanup()


if __name__ == "__main__":
    main()
