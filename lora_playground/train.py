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

from .optim import OPTIMIZER_CHOICES, build_optimizer, optimizer_config_dict
from .ucv_layer import inject_ucv_adapters
from .utils import collect_dense_target_weights, freeze_all_except_targets


TRAINING_MODES = ("lora", "svd_step_oracle", "svd_cumulative_oracle", "galore", "ucv")


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
    parser.add_argument("--max_train_samples", type=int, default=8000)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--data_dir", default=None, help="Pre-tokenized dataset dir (Arrow). Skips download + tokenization.")
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
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
    parser.add_argument("--precond_gamma", type=float, default=0.5,
                        help="Fractional power for PSI-LoRA/KFAC-LoRA K-FAC scaling.")
    parser.add_argument("--precond_ema_beta", type=float, default=0.99,
                        help="EMA smoothing for PSI-LoRA/KFAC-LoRA K-FAC statistics.")
    parser.add_argument("--precond_delta", type=float, default=1e-5,
                        help="Damping floor for PSI-LoRA/KFAC-LoRA K-FAC statistics.")
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
    parser.add_argument("--galore_update_proj_gap", type=int, default=200,
                        help="Steps between GaLore projection updates.")
    parser.add_argument("--galore_scale", type=float, default=0.25,
                        help="GaLore update scale factor.")
    parser.add_argument("--log_optim_diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="For diagnostic-emitting optimizers (adam-lin-lora, adam-scaled-lora, "
                             "adam-polar-product-lora, etc.): emit per-step JSONL `optim_step` events "
                             "with cosine(precond step, raw-Adam step), update norms, and S_A/S_B "
                             "eigenvalue extremes. ON by default (~10%% wall) — pass "
                             "--no-log_optim_diagnostics to disable for throughput-only benchmarks.")
    parser.add_argument("--optim_diagnostics_every", type=int, default=20,
                        help="Cadence (in optimizer steps) for --log_optim_diagnostics.")
    parser.add_argument("--debug_higham_residual", action="store_true",
                        help="Debug-only: every higham `_spd_inv_half` call emits a JSONL "
                             "`higham_residual` event with ‖Z H Z − I‖_F per matrix and "
                             "presence of non-finite output. Used to diagnose higham failures "
                             "(NaN at high r, drift) post-mortem. Cheap (~5%% wall on diagnostic "
                             "cadence). Off by default.")
    parser.add_argument("--precond_refresh_every", type=int, default=1,
                        help="K-step cadence for refreshing the per-pair Gram-preconditioner cache "
                             "(adam-scaled-lora, adam-lin-lora, adam-polar-product-lora, "
                             "adamuon-polar-product-lora). K=1 reproduces the original per-step "
                             "behavior; K>1 reuses the cached preconditioner for K-1 steps after "
                             "each refresh, trading a small amount of staleness for a large step-time "
                             "speedup at high LoRA rank.")
    parser.add_argument("--precond_method", choices=["eigh", "higham"], default="eigh",
                        help="Method for computing S^{-1/2} in the polar-product optimizers. "
                             "'eigh' is the reference (eigendecomp + diag-pow + reconstruct). "
                             "'higham' uses Newton-Schulz iteration (matmul-only) — much faster "
                             "at high LoRA rank because it avoids the eigh kernel-launch storm.")
    parser.add_argument("--higham_iters", type=int, default=10,
                        help="Newton-Schulz iterations when --precond_method=higham. "
                             "10 is needed for κ ≈ 200 (the worst case observed for SB "
                             "during training); 5 is fine on well-conditioned SA only.")
    parser.add_argument("--picard_alpha", type=float, default=1.0,
                        help="Damping on the Picard cross-coupling correction in "
                             "AdamPolarProductLoRA (only takes effect when picard_iters > 1). "
                             "α=1 standard Picard; α=0 zeros the cross-term; intermediate "
                             "values continuously interpolate between block-diagonal and "
                             "joint-NE targets.")
    parser.add_argument("--picard_iters_override", type=int, default=None,
                        help="Override picard_iters for AdamPolarProductLoRA "
                             "(adam-polar-product-lora-coupled). Default uses the "
                             "factory's hardcoded value (3 for coupled).")
    parser.add_argument("--polar_core_remix_alpha", type=float, default=0.0,
                        help="Core-signal remix coefficient. α=0 (default): no "
                             "remix. α=1/4: completed-core metric prediction "
                             "(attenuates agreed mode S_+ by half, preserves "
                             "disagreement mode S_-). Applied before Picard / "
                             "polar pipeline; replaces row(A) / col(B) "
                             "projections of (u_A, u_B) with remixed versions.")
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
                        choices=["ns", "ns_hybrid", "polar_express"],
                        help="Polar approximation method in adam-polar-product-lora's _polar_pipeline. "
                             "'ns' = standard degree-3 Newton-Schulz (default). "
                             "'ns_hybrid' = DeepSeek-V4 §2.4 two-stage degree-5 (8 aggressive + 2 refine). "
                             "'polar_express' = Amsel et al. arXiv:2505.16932 per-iteration optimal degree-5.")
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
    return parser


def main():
    args = make_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    if args.debug_higham_residual:
        from lora_playground import utils as _utils
        _utils.HIGHAM_DEBUG["enabled"] = True

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

    # Single-pass invariant: all training in this project must be one epoch
    # or less. Multi-epoch sweeps mix capacity-to-fit-the-subset effects with
    # per-step optimization quality, which makes optimizer comparisons
    # uninterpretable. (See: 8k-step runs invalidated 2026-05-04 because they
    # ran ~3.5 epochs over the 32k-sample subset.) Compute the total samples
    # the training loop will consume and refuse to start if it exceeds the
    # dataset. Override with --allow_multi_epoch only for explicit experiments.
    samples_consumed = (
        args.max_steps * args.batch_size * args.grad_accum_steps
    )
    if samples_consumed > len(train_dataset) and not getattr(args, "allow_multi_epoch", False):
        raise ValueError(
            f"Multi-epoch training blocked: max_steps × batch_size × grad_accum "
            f"= {args.max_steps} × {args.batch_size} × {args.grad_accum_steps} = "
            f"{samples_consumed:,} samples, but train dataset has only "
            f"{len(train_dataset):,} samples "
            f"(~{samples_consumed / len(train_dataset):.2f} epochs). "
            f"Either reduce --max_steps, increase --max_train_samples / dataset "
            f"size, or pass --allow_multi_epoch if multi-epoch is the intent."
        )

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
    ucv_target_names: list[str] = []
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
    elif args.training_mode == "ucv":
        model.to(device)
        ucv_injected = inject_ucv_adapters(
            model,
            target_modules=parsed_target_modules,
            r=args.lora_r,
            alpha=args.lora_alpha,
        )
        ucv_target_names = [name for name, _ in ucv_injected]
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
    optimizer = build_optimizer(
        model,
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
        muon_alpha=args.lora_alpha,
        muon_rank=args.lora_r,
        log_optim_diagnostics=args.log_optim_diagnostics,
        optim_diagnostics_every=args.optim_diagnostics_every,
        precond_refresh_every=args.precond_refresh_every,
        precond_method=args.precond_method,
        higham_iters=args.higham_iters,
        picard_alpha=args.picard_alpha,
        picard_iters_override=args.picard_iters_override,
        anderson_m=args.anderson_m,
        anderson_reg=args.anderson_reg,
        soap_beta=args.soap_beta,
        soap_refresh_every=args.soap_refresh_every,
        polar_norm_dir=args.polar_norm_dir,
        polar_sigma_power=args.polar_sigma_power,
        polar_method=args.polar_method,
        polar_core_remix_alpha=args.polar_core_remix_alpha,
        beta1=args.beta1,
        beta2=args.beta2,
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
            "profile_steps": args.profile_steps,
            "picard_alpha": args.picard_alpha,
            "picard_iters_override": args.picard_iters_override,
            "anderson_m": args.anderson_m,
            "anderson_reg": args.anderson_reg,
            "soap_beta": args.soap_beta,
            "soap_refresh_every": args.soap_refresh_every,
            "polar_norm_dir": args.polar_norm_dir,
            "polar_sigma_power": args.polar_sigma_power,
            "polar_method": args.polar_method,
            "polar_core_remix_alpha": args.polar_core_remix_alpha,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "optimizer_config": optimizer_config_dict(optimizer),
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
    # Windowed train-loss accumulator for `train_step` events.
    window_loss_sum = 0.0
    window_tokens = 0
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
