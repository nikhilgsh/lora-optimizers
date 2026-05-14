"""Shared model-construction kernel used by both `train.py` and the
`scripts/bench/bench_optimizer_step.py` end-to-end timing harness.

The duplication that hurt previously lived here: attn_implementation
fallback, PEFT/UCV/SVD-oracle wrap, gradient checkpointing, bare-model
snapshot for `collect_lora_pairs`, DDP wrap order, and torch.compile
ordering. Centralizing it means features like packed_v1 mask plumbing,
MFU constants, and DDP wrap-order changes only have to land once.

What is NOT in the kernel (deliberately):
- Tokenizer / dataset / collator / sampler / dataloader construction —
  bench uses synthetic batches, train uses the real dataset.
- Optimizer construction — `lora_playground.optim.build_optimizer`
  is already the sole factory; both call sites use it directly with
  different kwarg subsets.
- `scheduler`, `wandb`, `profiler`, eval cadence, JSONL logging.
- Bench's specialized phase-timed step harness.

The train-only `run_one_train_step` here mirrors the inner grad-accum
loop in `train.py::main` so a single change to the per-step path
(e.g. zero_grad semantics, clip ordering) lands in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from .ucv_layer import inject_ucv_adapters
from .utils import collect_dense_target_weights, freeze_all_except_targets


def _resolve_auto_attn() -> str:
    if not torch.cuda.is_available():
        return "sdpa"
    major = torch.cuda.get_device_capability()[0]
    if major >= 9:
        return "flash_attention_4"
    if major == 8:
        return "flash_attention_2"
    return "sdpa"


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def count_tokens(batch):
    return int((batch["labels"] != -100).sum().item())


def _apply_liger(model_name: str, fused_linear_ce: bool = False) -> str | None:
    """Family-dispatch Liger Kernel patches. Must be called BEFORE
    from_pretrained. Returns the patched family name."""
    name_lower = model_name.lower()
    kw = {"fused_linear_cross_entropy": fused_linear_ce}
    if "olmo" in name_lower:
        from liger_kernel.transformers import apply_liger_kernel_to_olmo2
        apply_liger_kernel_to_olmo2(**kw)
        return "olmo2"
    if "llama" in name_lower:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama(**kw)
        return "llama"
    if "qwen3" in name_lower:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(**kw)
        return "qwen3"
    if "qwen2" in name_lower:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2(**kw)
        return "qwen2"
    raise ValueError(
        f"--use_liger requested but no Liger family matches model_name={model_name!r}. "
        f"Supported substrings: olmo, llama, qwen2, qwen3."
    )


@dataclass
class PeftModel:
    """Outputs of build_peft_model.

    bare_model: PEFT/UCV/freeze + (optional) torch.compile, NEVER
                DDP-wrapped. This is what custom optimizers walk via
                collect_lora_pairs / collect_dense_target_weights.
    train_model: the model the forward pass goes through. Same as
                 bare_model in single-process mode; DDP-wrapped in
                 distributed mode.
    """
    bare_model: nn.Module
    train_model: nn.Module
    liger_applied: str | None
    resolved_attn: str
    dense_targets: list = field(default_factory=list)
    ucv_target_names: list[str] = field(default_factory=list)


def build_peft_model(
    *,
    model_name: str,
    training_mode: str = "lora",
    target_modules,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    dtype: torch.dtype | None,
    attn_implementation: str | None = "sdpa",
    use_liger: bool = False,
    liger_flce: bool = False,
    gradient_checkpointing: bool = False,
    compile_mode: str | None = None,
    device: torch.device,
    world_size: int = 1,
    local_rank: int = 0,
) -> PeftModel:
    """Single source of truth for model construction.

    Steps (in order):
      1. Liger Kernel patch (if requested) — must run BEFORE from_pretrained.
      2. AutoModelForCausalLM.from_pretrained with attn_implementation
         fallback chain (flash_attention_4 → flash_attention_2 → sdpa).
      3. use_cache=False (training never wants KV cache).
      4. PEFT wrap / UCV inject / SVD-oracle freeze, dispatched on
         training_mode.
      5. .to(device).
      6. gradient_checkpointing (after PEFT, before DDP/compile).
      7. Snapshot `bare_model` reference (the un-DDP-un-compile model).
      8. DDP wrap (if world_size > 1) — BEFORE compile (PyTorch 2.x
         recommended order).
      9. torch.compile (if compile_mode is not None).

    `compile_mode=None` means no compile. `compile_mode="default"` runs
    torch.compile with default mode. Other strings forward to
    torch.compile(mode=...).
    """
    # 1. Liger patching MUST happen before from_pretrained.
    liger_applied = None
    if use_liger:
        liger_applied = _apply_liger(model_name, fused_linear_ce=liger_flce)
        print(f"# Liger Kernel applied: family={liger_applied}", flush=True)

    # 2. Resolve attn_implementation, with auto-detect + fallback chain.
    # `attn_implementation=None` skips the key entirely (HF default).
    model_kwargs = {"dtype": dtype} if dtype is not None else {}
    attn_impl = attn_implementation
    if attn_impl == "auto":
        attn_impl = _resolve_auto_attn()
        print(f"# attn_implementation=auto resolved to {attn_impl}", flush=True)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    fallback_chain = {
        "flash_attention_4": "flash_attention_2",
        "flash_attention_2": "sdpa",
    }
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            break
        except (ImportError, ValueError) as e:
            cur = model_kwargs.get("attn_implementation")
            nxt = fallback_chain.get(cur) if cur is not None else None
            if nxt is None:
                raise
            print(f"# {cur} unavailable ({e}); falling back to {nxt}", flush=True)
            model_kwargs["attn_implementation"] = nxt

    # 3.
    model.config.use_cache = False

    # 4. + 5. PEFT / UCV / SVD-oracle dispatch + .to(device).
    dense_targets: list = []
    ucv_target_names: list[str] = []
    if training_mode == "lora":
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.to(device)
    elif training_mode == "ucv":
        model.to(device)
        ucv_injected = inject_ucv_adapters(
            model,
            target_modules=target_modules,
            r=lora_r,
            alpha=lora_alpha,
        )
        ucv_target_names = [name for name, _ in ucv_injected]
    else:
        # svd_step_oracle / svd_cumulative_oracle / galore — unfreeze
        # dense targets, freeze the rest. Optimizer takes dense_targets
        # via build_optimizer kwargs.
        model.to(device)
        dense_targets = collect_dense_target_weights(model, target_modules)
        freeze_all_except_targets(model, dense_targets)

    # 6.
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # 7. Save a reference to the un-DDP-un-compile-wrapped model. The
    # custom optimizers walk model.named_modules() to find lora_A/lora_B
    # parameters; passing the bare PEFT model avoids needing to pierce
    # DDP/compile wrappers later. The actual training forward/backward
    # still goes through the DDP+compile wrappers below.
    bare_model = model

    # 8. DDP BEFORE compile (recommended order in PyTorch 2.x). DDP
    # installs gradient hooks that all-reduce param.grad during backward;
    # this keeps Adam-family optimizer state synchronized across ranks
    # by construction.
    train_model = bare_model
    if world_size > 1:
        ddp_kwargs = (
            {"device_ids": [local_rank], "output_device": local_rank}
            if device.type == "cuda" else {}
        )
        train_model = DDP(bare_model, **ddp_kwargs)

    # 9.
    if compile_mode is not None:
        compile_kwargs = {}
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        train_model = torch.compile(train_model, **compile_kwargs)

    # PeftModel proxies .config to the base HF model; bare HF models have
    # .config directly. Both paths expose ._attn_implementation.
    resolved_attn = getattr(bare_model.config, "_attn_implementation", attn_impl)
    return PeftModel(
        bare_model=bare_model,
        train_model=train_model,
        liger_applied=liger_applied,
        resolved_attn=resolved_attn,
        dense_targets=dense_targets,
        ucv_target_names=ucv_target_names,
    )


def run_one_train_step(
    train_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_iter: Iterator,
    train_loader,
    *,
    grad_accum_steps: int,
    max_grad_norm: float | None,
    device: torch.device,
) -> tuple[float, int, Iterator, dict]:
    """One macro-step of training (train.py only).

    Sequence:
      zero_grad(set_to_none=True)
      → grad_accum × (next-batch, fwd, scaled bwd), skipping batches with
        zero supervised tokens
      → optional grad-clip
      → optimizer.step()

    Returns (step_loss_weighted_sum, step_tokens, train_iter). The iter
    is returned because StopIteration triggers a fresh `iter(train_loader)`
    that the caller needs to thread back through subsequent steps.

    Caller is responsible for: scheduler.step(), profiler.step(),
    eval cadence, JSONL logging, wandb. Bench does NOT use this helper —
    its specialized timing harness is in scripts/bench/bench_optimizer_step.py.
    """
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    step_tokens = 0
    skipped_zero_token_microbatches = 0
    for _ in range(grad_accum_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch_to_device(batch, device)
        tokens = count_tokens(batch)
        if tokens == 0:
            # Cross-entropy over an all-ignored-label batch is NaN in HF
            # causal-LM heads. It contributes no supervised objective, so skip
            # the forward/backward pass instead of poisoning the loss window.
            skipped_zero_token_microbatches += 1
            continue
        outputs = train_model(**batch)
        (outputs.loss / grad_accum_steps).backward()
        step_loss += float(outputs.loss.detach()) * tokens
        step_tokens += tokens

    if step_tokens == 0:
        raise RuntimeError(
            "All gradient-accumulation microbatches had zero supervised tokens; "
            "cannot take an optimizer step."
        )

    # Pre-step global norm probe — covers every optimizer (param + raw grad
    # L2, count of non-finite grad entries) regardless of optimizer family.
    # Caller decides whether to emit the returned dict; computation is cheap
    # (one foreach_norm pass over trainable params).
    norm_stats = _compute_global_norms(train_model)
    if skipped_zero_token_microbatches:
        norm_stats = dict(norm_stats)
        norm_stats["skipped_zero_token_microbatches"] = skipped_zero_token_microbatches

    if max_grad_norm and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)
    optimizer.step()
    return step_loss, step_tokens, train_iter, norm_stats


def _compute_global_norms(model: nn.Module) -> dict:
    """Global L2 norm of trainable params and their gradients, plus a
    count of non-finite grad entries. Returns a dict (may be empty if
    no trainable params have grads, e.g. the very first step before
    backward). Cost: one foreach pass over trainable params; microseconds
    on LoRA-sized parameter sets."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {}
    grads = [p.grad for p in params if p.grad is not None]
    out = {
        "param_l2": float(torch.norm(torch.stack(
            [p.detach().float().norm() for p in params]))),
        "n_params_with_grad": len(grads),
    }
    if grads:
        out["grad_l2"] = float(torch.norm(torch.stack(
            [g.detach().float().norm() for g in grads])))
        # Count non-finite ENTRIES across all grad tensors, not the number
        # of grad tensors that contain any non-finite. (Earlier version
        # used .any() which under-reported the count by orders of magnitude
        # — capped at the number of trainable params, not entries.)
        n_nonfinite = int(sum(
            (~torch.isfinite(g)).sum().item() for g in grads))
        out["n_non_finite_grads"] = n_nonfinite
    return out
