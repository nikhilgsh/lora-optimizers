"""Sequence-packed train collator and pad-to-max eval collator.

Implements the `packed_v1` data pipeline. Train-side examples are packed
offline (during tokenization) into fixed-length slots; the train collator
then just stacks pre-packed rows and builds the per-batch 4D SDPA mask.
Eval-side examples stay unpacked but are padded to `max_seq_length` so
shapes are static for `torch.compile`.

Why this shape:
  - Static shapes ⇒ torch.compile recompiles once.
  - Train: ~3× more signal tokens / GPU-hour at seq=2048 on Magicoder.
  - Eval: per-doc CE retains the standard "loss per response token"
    semantics needed for optimizer comparisons.

SDPA-only (no FlashAttention-2 cu_seq_lens shortcut). Project profile
showed SDPA ≈ FA-2 walltime; we don't pay the FA-2 integration cost.

Schema (per row, after offline packing):
  - input_ids:    list[int],  length == seq_length  (pad_token on tail)
  - labels:       list[int],  length == seq_length  (-100 on prompt + pad)
  - position_ids: list[int],  length == seq_length  (resets per doc)
  - doc_lens:     list[int],  per-doc lengths within this slot
                              (sum may be < seq_length if pad-tailed)

See docs/notes/polar_product/data_pipeline_followups.md for the design
context and the verification plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# ─── primitive helpers (pure-tensor, easy to test) ────────────────────────────


def build_block_diagonal_causal_mask(
    doc_lens: list[int],
    seq_length: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Construct a (1, 1, seq_length, seq_length) additive SDPA mask:
    zero on positions allowed by intra-doc causality, -inf everywhere else.

    Pad-tail positions (sum(doc_lens) < seq_length) are left at -inf;
    the loss is -100 there so the rows don't contribute to gradients.
    """
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
    mask = torch.full(
        (seq_length, seq_length), neg_inf, dtype=dtype, device=device
    )
    start = 0
    for length in doc_lens:
        if length <= 0:
            continue
        end = min(start + length, seq_length)
        # Causal block: zero on/below diagonal, -inf above.
        intra = torch.triu(
            torch.full((end - start, end - start), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[start:end, start:end] = intra
        start = end
        if start >= seq_length:
            break
    return mask[None, None, :, :]


def build_packed_position_ids(
    doc_lens: list[int],
    seq_length: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-doc position_ids restart at 0; pad tail counts up from 0
    (its values don't matter — masked out by loss + attention)."""
    out = torch.zeros(seq_length, dtype=torch.long, device=device)
    start = 0
    for length in doc_lens:
        if length <= 0:
            continue
        end = min(start + length, seq_length)
        out[start:end] = torch.arange(end - start, dtype=torch.long, device=device)
        start = end
        if start >= seq_length:
            break
    if start < seq_length:
        out[start:seq_length] = torch.arange(
            seq_length - start, dtype=torch.long, device=device
        )
    return out


# ─── offline packing ──────────────────────────────────────────────────────────


@dataclass
class _Doc:
    input_ids: list[int]
    prompt_len: int


def pack_documents(
    docs: list[_Doc] | list[dict],
    seq_length: int,
    pad_token_id: int,
) -> list[dict]:
    """Greedy first-fit pack a stream of tokenized documents into
    `seq_length`-wide slots. Each slot is a dict with input_ids, labels,
    position_ids, doc_lens (all already at length seq_length except
    doc_lens which is variable).

    Docs longer than seq_length after tokenization should already have
    been truncated upstream; we assert here defensively.
    """
    out: list[dict] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_pos: list[int] = []
    cur_doc_lens: list[int] = []

    def flush():
        nonlocal cur_ids, cur_labels, cur_pos, cur_doc_lens
        if not cur_ids:
            return
        n_pad = seq_length - len(cur_ids)
        ids = cur_ids + [pad_token_id] * n_pad
        labels = cur_labels + [-100] * n_pad
        pos = cur_pos + list(range(n_pad))
        out.append({
            "input_ids": ids,
            "labels": labels,
            "position_ids": pos,
            "doc_lens": list(cur_doc_lens),
        })
        cur_ids, cur_labels, cur_pos, cur_doc_lens = [], [], [], []

    for d in docs:
        if isinstance(d, dict):
            ids = list(d["input_ids"])
            prompt_len = int(d.get("prompt_len", 0))
        else:
            ids = list(d.input_ids)
            prompt_len = int(d.prompt_len)
        L = len(ids)
        if L == 0:
            continue
        if L > seq_length:
            raise ValueError(
                f"Document of length {L} exceeds seq_length={seq_length}; "
                f"truncate upstream."
            )
        if len(cur_ids) + L > seq_length:
            flush()
        # Append.
        cur_ids.extend(ids)
        # labels: -100 on prompt positions, real ids on response positions
        labels = [-100] * min(prompt_len, L) + ids[prompt_len:]
        # If prompt_len > L (shouldn't happen post-truncation but be safe),
        # all positions are masked.
        if len(labels) != L:
            labels = labels[:L]
        cur_labels.extend(labels)
        cur_pos.extend(range(L))
        cur_doc_lens.append(L)
    flush()
    return out


# ─── collators ────────────────────────────────────────────────────────────────


class PackingCollator:
    """Stacks pre-packed rows into a batch and constructs the per-batch
    4D SDPA attention_mask from each row's `doc_lens`.

    Input rows are emitted by `pack_documents`. Each row has identical
    seq_length but a per-row `doc_lens` list — so the 4D mask has to
    be batched (B, 1, S, S), one block-diagonal pattern per row.
    """

    def __init__(self, seq_length: int, mask_dtype: torch.dtype = torch.float32):
        self.seq_length = seq_length
        self.mask_dtype = mask_dtype

    def __call__(self, features: list[dict]) -> dict:
        S = self.seq_length
        B = len(features)
        input_ids = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        labels = torch.tensor(
            [f["labels"] for f in features], dtype=torch.long
        )
        position_ids = torch.tensor(
            [f["position_ids"] for f in features], dtype=torch.long
        )
        # Build per-row 4D mask. We stack on dim 0 to (B, 1, S, S).
        masks = [
            build_block_diagonal_causal_mask(
                list(f["doc_lens"]), S, self.mask_dtype, "cpu"
            )
            for f in features
        ]
        attention_mask = torch.cat(masks, dim=0)  # (B, 1, S, S)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


class PadToMaxCollator:
    """Eval collator: pad each (variable-length) tokenized doc to
    `seq_length`, build standard 2D attention_mask, apply prompt-mask
    via labels. Static shape ⇒ compile-friendly; per-doc CE semantics
    are preserved (one doc per row, no cross-doc attention).
    """

    def __init__(self, seq_length: int, pad_token_id: int):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        S = self.seq_length
        B = len(features)
        input_ids = torch.full((B, S), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, S), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, S), dtype=torch.long)
        for i, f in enumerate(features):
            ids = list(f["input_ids"])
            prompt_len = int(f.get("prompt_len", 0))
            L = min(len(ids), S)
            input_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            attention_mask[i, :L] = 1
            # labels: -100 on prompt, real ids on response, -100 on pad.
            real_start = min(prompt_len, L)
            if real_start < L:
                labels[i, real_start:L] = torch.tensor(
                    ids[real_start:L], dtype=torch.long
                )
        position_ids = (
            torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S).clone()
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
