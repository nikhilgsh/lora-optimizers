import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)

from .optim import OPTIMIZER_CHOICES, build_optimizer
from .utils import collect_dense_target_weights, freeze_all_except_targets


TRAINING_MODES = ("lora", "svd_step_oracle", "svd_cumulative_oracle")


def format_example(example):
    if "prompt" in example and "completion" in example:
        return f"{example['prompt'].strip()}\n{example['completion'].strip()}"
    if "instruction" in example and "response" in example:
        return f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
    if "prompt" in example and "response" in example:
        return f"{example['prompt'].strip()}\n{example['response'].strip()}"
    if "prompt" in example and isinstance(example["prompt"], str):
        return example["prompt"]
    if {"instruction", "input", "output"}.issubset(example):
        pieces = [f"Instruction:\n{example['instruction']}"]
        if str(example["input"]).strip():
            pieces.append(f"Input:\n{example['input']}")
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


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def count_tokens(batch):
    return int((batch["labels"] != -100).sum().item())


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, dataloader, device):
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
    return total_loss / max(total_tokens, 1)


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


def log_event(payload):
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
    parser.add_argument("--max_train_samples", type=int, default=4096)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--data_dir", default=None, help="Pre-tokenized dataset dir (Arrow). Skips download + tokenization.")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", default="constant")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--svd_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--lora_plus_multiplier", type=float, default=1.0)
    parser.add_argument("--scaled_metric", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--profile_dir", default="runs/profiles")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--target_eval_loss", type=float, default=None)
    parser.add_argument("--wandb_project", default=None, help="W&B project name. Omit to disable W&B.")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name. Auto-generated from key params if omitted.")
    return parser


def main():
    args = make_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
        train_dataset, eval_dataset = tokenize_splits(train_raw, eval_raw, tokenizer, args)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    parsed_target_modules = parse_target_modules(args.target_modules)
    svd_rank = args.svd_rank if args.svd_rank is not None else args.lora_r

    model_kwargs = {"dtype": dtype} if dtype is not None else {}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    dense_targets = []
    if args.training_mode == "lora":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parsed_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.to(device)
    else:
        model.to(device)
        dense_targets = collect_dense_target_weights(model, parsed_target_modules)
        freeze_all_except_targets(model, dense_targets)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.compile:
        compile_kwargs = {}
        if args.compile_mode != "default":
            compile_kwargs["mode"] = args.compile_mode
        model = torch.compile(model, **compile_kwargs)

    if args.training_mode == "lora":
        if args.optimizer in {"svd-step-adamw", "svd-cumulative-adamw"}:
            raise ValueError("SVD optimizers require --training_mode svd_step_oracle or svd_cumulative_oracle.")
        effective_optimizer = args.optimizer
        rank_constraint = None
    elif args.training_mode == "svd_step_oracle":
        if args.optimizer not in {"adamw", "svd-step-adamw"}:
            raise ValueError("svd_step_oracle currently supports AdamW only.")
        effective_optimizer = "svd-step-adamw"
        rank_constraint = "per_step_update"
    else:
        if args.optimizer not in {"adamw", "svd-cumulative-adamw"}:
            raise ValueError("svd_cumulative_oracle currently supports AdamW only.")
        effective_optimizer = "svd-cumulative-adamw"
        rank_constraint = "cumulative_displacement"

    optimizer = build_optimizer(
        model,
        optimizer_type=effective_optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scaled_metric=args.scaled_metric,
        lora_plus_multiplier=args.lora_plus_multiplier,
        targets=dense_targets if dense_targets else None,
        svd_rank=svd_rank if dense_targets else None,
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    wandb_run = None
    if args.wandb_project:
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

    log_event(
        {
            "event": "config",
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "git_commit": git_commit(),
            "device": str(device),
            "training_mode": args.training_mode,
            "optimizer": effective_optimizer,
            "requested_optimizer": args.optimizer,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name if not args.train_file else "json",
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "svd_rank": svd_rank if dense_targets else None,
            "rank_constraint": rank_constraint,
            "target_module_count": len(dense_targets),
            "target_module_names": [target.name for target in dense_targets],
            "svd_projection": "exact" if dense_targets else None,
            "exclude_lm_head_from_all_linear": bool(dense_targets and parsed_target_modules == "all-linear"),
            "seed": args.seed,
            "bf16": use_bf16,
            "tf32": device.type == "cuda" and not args.no_tf32,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "profile_steps": args.profile_steps,
        }
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

    train_iter = iter(train_loader)
    total_tokens = 0
    eval_elapsed = 0.0
    cuda_sync()
    start = time.perf_counter()
    model.train()

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch_to_device(batch, device)
            tokens = count_tokens(batch)
            outputs = model(**batch)
            (outputs.loss / args.grad_accum_steps).backward()
            step_loss += float(outputs.loss.detach()) * tokens
            step_tokens += tokens

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_tokens += step_tokens
        if profiler is not None:
            profiler.step()

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
            eval_payload = {
                "event": "eval",
                "step": step,
                "train_loss": step_loss / max(step_tokens, 1),
                "eval_loss": eval_loss,
                "tokens": total_tokens,
                "train_elapsed_sec": train_elapsed,
                "eval_sec": eval_sec,
                "tokens_per_sec": total_tokens / max(train_elapsed, 1e-9),
                "peak_memory_mb": peak_memory_mb,
                "lr": scheduler.get_last_lr()[0],
            }
            log_event(eval_payload)
            if wandb_run is not None:
                wandb_run.log(
                    {k: v for k, v in eval_payload.items() if k not in ("event",)},
                    step=step,
                )
            if args.target_eval_loss is not None and eval_loss <= args.target_eval_loss:
                break

    if profiler is not None:
        profiler.stop()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
