"""Tests for `lora_playground.data` — the packed_v1 data pipeline.

The behavioral-equivalence tests (single-doc and two-doc reductions) are
the load-bearing piece: they prove the custom 4D SDPA mask + per-doc
position_ids reset reproduces the standard unpacked numerics. Without
these, any silent attention leak across doc boundaries would confound
every downstream optimizer comparison.

All tests are CPU-only and use a tiny randomly-initialized HF causal LM
(LlamaForCausalLM with d_model=64, 2 layers) so `pytest -q` stays fast.
"""
from __future__ import annotations

import random

import pytest
import torch

from lora_playground.data import (
    PackingCollator,
    PadToMaxCollator,
    build_block_diagonal_causal_mask,
    build_packed_position_ids,
    pack_documents,
)


# ─── tiny HF model fixture ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_llama():
    """Tiny randomly-initialized Llama for CPU equivalence tests."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


# ─── 1. shape constancy ───────────────────────────────────────────────────────


def _make_doc(L, prompt_len=0, vocab=64, seed=0):
    rng = random.Random(seed)
    ids = [rng.randint(2, vocab - 1) for _ in range(L)]
    return {"input_ids": ids, "prompt_len": prompt_len}


def test_pack_documents_drops_zero_supervision_slot():
    """packed_v1.1 contract: a slot consisting entirely of prompt-masked
    tokens (every label = -100) is discarded, because such a slot would
    compute NaN cross-entropy in HF's masked CE."""
    seq_length = 16
    pad_id = 0
    # Two docs whose prompts fill them entirely, packed alone or together
    # into a single slot — every position is masked.
    all_prompt = [_make_doc(L=8, prompt_len=8, seed=1),
                  _make_doc(L=6, prompt_len=6, seed=2)]
    # Plus one healthy doc to verify good slots survive.
    healthy = [_make_doc(L=10, prompt_len=2, seed=3)]
    docs = all_prompt + healthy
    packs = pack_documents(docs, seq_length=seq_length, pad_token_id=pad_id,
                           drop_zero_supervision_slots=True)
    # Healthy slot must survive; the all-prompt slot(s) must be dropped.
    assert len(packs) == 1, (
        f"expected only the healthy slot to survive; got {len(packs)} slots"
    )
    surviving = packs[0]
    assert any(l != -100 for l in surviving["labels"]), (
        "surviving slot has no supervised tokens — filter let a zero-sup slot through"
    )


def test_pack_documents_filter_opt_out():
    """Setting drop_zero_supervision_slots=False preserves legacy
    behavior — all-prompt slots are emitted unchanged."""
    seq_length = 16
    pad_id = 0
    docs = [_make_doc(L=8, prompt_len=8, seed=1)]
    packs_kept = pack_documents(docs, seq_length=seq_length, pad_token_id=pad_id,
                                drop_zero_supervision_slots=False)
    assert len(packs_kept) == 1, "opt-out path should keep the all-prompt slot"
    assert all(l == -100 for l in packs_kept[0]["labels"]), (
        "all-prompt slot should have every label = -100"
    )


def test_pack_documents_shape_constancy():
    seq_length = 16
    pad_id = 0
    docs = [_make_doc(L=5, seed=i) for i in range(7)]
    packs = pack_documents(docs, seq_length=seq_length, pad_token_id=pad_id)
    assert len(packs) >= 1
    for p in packs:
        assert len(p["input_ids"]) == seq_length
        assert len(p["labels"]) == seq_length
        assert len(p["position_ids"]) == seq_length
    collator = PackingCollator(seq_length=seq_length)
    batch = collator(packs)
    B = len(packs)
    assert batch["input_ids"].shape == (B, seq_length)
    assert batch["labels"].shape == (B, seq_length)
    assert batch["position_ids"].shape == (B, seq_length)
    assert batch["attention_mask"].shape == (B, 1, seq_length, seq_length)


# ─── 2. mask binary structure ─────────────────────────────────────────────────


def _is_blocked(x: float) -> bool:
    """Match HF convention: blocked = -inf OR finfo.min (some SDPA kernels
    misbehave with literal -inf, so HF uses finfo.min for additive masks)."""
    return x == float("-inf") or x <= torch.finfo(torch.float32).min / 2


def test_block_diagonal_mask_structure():
    seq = 10
    L1, L2 = 4, 3
    mask = build_block_diagonal_causal_mask([L1, L2], seq, torch.float32, "cpu")
    M = mask[0, 0]  # (seq, seq)
    # Within first doc: zero on/below diagonal, blocked above.
    for q in range(L1):
        for k in range(L1):
            if k <= q:
                assert M[q, k].item() == 0.0, (q, k)
            else:
                assert _is_blocked(M[q, k].item()), (q, k)
    # Cross-doc: position L1 (start of doc2) cannot see any of doc1.
    for k in range(L1):
        assert _is_blocked(M[L1, k].item()), k
    # Within doc2 (offset by L1), causal triangle.
    for q in range(L2):
        for k in range(L2):
            ent = M[L1 + q, L1 + k].item()
            if k <= q:
                assert ent == 0.0, (q, k)
            else:
                assert _is_blocked(ent), (q, k)
    # Pad-tail rows / cols all blocked.
    for q in range(L1 + L2, seq):
        for k in range(seq):
            assert _is_blocked(M[q, k].item()), (q, k)


# ─── 3. position-ids reset ────────────────────────────────────────────────────


def test_packed_position_ids_reset():
    seq = 10
    pos = build_packed_position_ids([4, 3], seq).tolist()
    assert pos[:4] == [0, 1, 2, 3]
    assert pos[4:7] == [0, 1, 2]
    # Pad tail counts up from 0; values don't matter for correctness, but
    # check we're at least returning something deterministic.
    assert pos[7:] == [0, 1, 2]


# ─── 4. prompt-mask placement ─────────────────────────────────────────────────


def test_pack_documents_prompt_mask_placement():
    seq = 12
    pad_id = 0
    doc = {"input_ids": [10, 11, 12, 13, 14, 15], "prompt_len": 3}
    packs = pack_documents([doc], seq_length=seq, pad_token_id=pad_id)
    assert len(packs) == 1
    p = packs[0]
    assert p["input_ids"][:6] == [10, 11, 12, 13, 14, 15]
    # First 3 positions: prompt (masked); next 3: response (real ids); pad tail.
    assert p["labels"][:3] == [-100, -100, -100]
    assert p["labels"][3:6] == [13, 14, 15]
    assert p["labels"][6:] == [-100] * 6
    assert p["doc_lens"] == [6]


# ─── 5-6. single-doc reduction (forward + backward) ───────────────────────────


def _run_unpacked(model, ids: list[int]):
    """Reference path: standard 2D mask, default position_ids."""
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = model(input_ids=input_ids, output_hidden_states=False)
    return out.logits  # (1, L, V)


def _run_packed(model, packed_row: dict, seq_length: int):
    """Custom 4D mask + reset position_ids path."""
    collator = PackingCollator(seq_length=seq_length, mask_dtype=torch.float32)
    batch = collator([packed_row])
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        output_hidden_states=False,
    )
    return out.logits  # (1, S, V)


def test_single_doc_reduction_forward(tiny_llama):
    seq = 16
    ids = [5, 7, 9, 11, 13, 15, 17, 19]  # length 8
    L = len(ids)
    # Pack as one doc into a single slot.
    packed = pack_documents(
        [{"input_ids": ids, "prompt_len": 0}],
        seq_length=seq,
        pad_token_id=0,
    )
    assert len(packed) == 1
    with torch.no_grad():
        ref = _run_unpacked(tiny_llama, ids)         # (1, L, V)
        got = _run_packed(tiny_llama, packed[0], seq)  # (1, S, V)
    # Compare on the L real positions only; pad tail is masked (-inf rows)
    # so its logits are noise.
    torch.testing.assert_close(got[:, :L], ref, rtol=1e-4, atol=1e-5)


def test_single_doc_reduction_backward(tiny_llama):
    seq = 16
    ids = [5, 7, 9, 11, 13, 15, 17, 19]
    L = len(ids)
    packed = pack_documents(
        [{"input_ids": ids, "prompt_len": 0}],
        seq_length=seq,
        pad_token_id=0,
    )

    def grads_for(logits):
        # Sum over real positions, backprop, return per-param grads.
        loss = logits.sum()
        for p in tiny_llama.parameters():
            if p.grad is not None:
                p.grad = None
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in tiny_llama.named_parameters()
                if p.grad is not None}

    # Reference: real positions only.
    ref_logits = _run_unpacked(tiny_llama, ids)  # (1, L, V)
    ref_grads = grads_for(ref_logits)

    # Packed: sum over real positions only (slice [:, :L]).
    packed_logits_full = _run_packed(tiny_llama, packed[0], seq)  # (1, S, V)
    packed_logits = packed_logits_full[:, :L]
    pack_grads = grads_for(packed_logits)

    # Same set of param names.
    assert set(ref_grads) == set(pack_grads)
    for name in ref_grads:
        torch.testing.assert_close(
            pack_grads[name], ref_grads[name], rtol=1e-4, atol=1e-5,
            msg=lambda m, n=name: f"grad mismatch on {n}: {m}",
        )


# ─── 7-8. two-doc independence (forward + backward) ──────────────────────────


def test_two_doc_independence_forward(tiny_llama):
    """Pack two distinct docs into one slot; per-doc logits should match
    running each doc through the model separately. This proves doc-aware
    attention works (no cross-doc leak)."""
    seq = 16
    ids_a = [3, 4, 5, 6, 7]
    ids_b = [11, 12, 13, 14, 15, 16]
    La, Lb = len(ids_a), len(ids_b)
    docs = [
        {"input_ids": ids_a, "prompt_len": 0},
        {"input_ids": ids_b, "prompt_len": 0},
    ]
    packed = pack_documents(docs, seq_length=seq, pad_token_id=0)
    assert len(packed) == 1
    assert packed[0]["doc_lens"] == [La, Lb]

    with torch.no_grad():
        ref_a = _run_unpacked(tiny_llama, ids_a)  # (1, La, V)
        ref_b = _run_unpacked(tiny_llama, ids_b)  # (1, Lb, V)
        got_full = _run_packed(tiny_llama, packed[0], seq)  # (1, S, V)
    got_a = got_full[:, :La]
    got_b = got_full[:, La:La + Lb]
    torch.testing.assert_close(got_a, ref_a, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_b, ref_b, rtol=1e-4, atol=1e-5)


def test_two_doc_independence_backward(tiny_llama):
    """Sum of independent single-doc grads must equal packed-grad."""
    seq = 16
    ids_a = [3, 4, 5, 6, 7]
    ids_b = [11, 12, 13, 14, 15, 16]
    La, Lb = len(ids_a), len(ids_b)
    docs = [
        {"input_ids": ids_a, "prompt_len": 0},
        {"input_ids": ids_b, "prompt_len": 0},
    ]
    packed = pack_documents(docs, seq_length=seq, pad_token_id=0)

    def zero_grads():
        for p in tiny_llama.parameters():
            if p.grad is not None:
                p.grad = None

    def collect_grads():
        return {n: p.grad.detach().clone() for n, p in tiny_llama.named_parameters()
                if p.grad is not None}

    # Reference: a + b grads from independent forwards.
    zero_grads()
    log_a = _run_unpacked(tiny_llama, ids_a)
    log_b = _run_unpacked(tiny_llama, ids_b)
    (log_a.sum() + log_b.sum()).backward()
    ref = collect_grads()

    # Packed: single forward, sum over both docs' positions.
    zero_grads()
    got_full = _run_packed(tiny_llama, packed[0], seq)  # (1, S, V)
    real = got_full[:, :La + Lb]
    real.sum().backward()
    got = collect_grads()

    assert set(ref) == set(got)
    for name in ref:
        torch.testing.assert_close(
            got[name], ref[name], rtol=1e-4, atol=1e-5,
            msg=lambda m, n=name: f"grad mismatch on {n}: {m}",
        )


# ─── 9. loss-masking equivalence ──────────────────────────────────────────────


def test_loss_masking_equivalence(tiny_llama):
    """Packed loss with prompt_len=k should equal manually-computed CE
    over response positions only on the unpacked single-doc forward."""
    import torch.nn.functional as F

    seq = 16
    ids = [3, 4, 5, 6, 7, 8, 9, 10]
    prompt_len = 3
    L = len(ids)
    packed = pack_documents(
        [{"input_ids": ids, "prompt_len": prompt_len}],
        seq_length=seq,
        pad_token_id=0,
    )
    p = packed[0]
    input_ids = torch.tensor([p["input_ids"]], dtype=torch.long)
    labels = torch.tensor([p["labels"]], dtype=torch.long)
    pos = torch.tensor([p["position_ids"]], dtype=torch.long)
    collator = PackingCollator(seq_length=seq)
    attn = collator([p])["attention_mask"]
    out = tiny_llama(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=pos,
        labels=labels,
    )
    packed_loss = out.loss

    # Reference: forward unpacked, manually compute CE over response
    # positions only (next-token prediction → shift by 1).
    ref_logits = _run_unpacked(tiny_llama, ids)  # (1, L, V)
    # Standard causal-LM shift: predict ids[i+1] from logits[..., i, :].
    shift_logits = ref_logits[:, :-1, :].contiguous()  # (1, L-1, V)
    shift_labels = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)  # (1, L-1)
    # Mask out any prediction whose target is in the prompt region.
    # Position i in shift_logits predicts ids[i+1]; mask if i+1 < prompt_len.
    keep_mask = torch.tensor(
        [(i + 1) >= prompt_len for i in range(L - 1)], dtype=torch.bool
    )
    flat_logits = shift_logits[0][keep_mask]
    flat_labels = shift_labels[0][keep_mask]
    ref_loss = F.cross_entropy(flat_logits, flat_labels)
    torch.testing.assert_close(packed_loss, ref_loss, rtol=1e-4, atol=1e-5)


# ─── 10. property test (random shapes) ────────────────────────────────────────


@pytest.mark.parametrize("seed", list(range(8)))
def test_pack_random_shapes_independence_forward(tiny_llama, seed):
    """Random packing of 1-3 docs; forward equivalence."""
    rng = random.Random(seed)
    seq = 24
    n_docs = rng.randint(1, 3)
    # Doc lengths sum to <= seq.
    remaining = seq
    docs_meta = []
    for _ in range(n_docs):
        if remaining < 2:
            break
        L = rng.randint(2, min(8, remaining))
        ids = [rng.randint(2, 60) for _ in range(L)]
        prompt_len = rng.randint(0, L - 1)
        docs_meta.append({"input_ids": ids, "prompt_len": prompt_len, "L": L})
        remaining -= L
    n_docs = len(docs_meta)
    if n_docs == 0:
        pytest.skip("trivial; nothing to pack")

    packed = pack_documents(
        [{"input_ids": d["input_ids"], "prompt_len": d["prompt_len"]} for d in docs_meta],
        seq_length=seq,
        pad_token_id=0,
    )
    # All docs should fit in one slot since their sum <= seq.
    assert len(packed) == 1

    with torch.no_grad():
        got = _run_packed(tiny_llama, packed[0], seq)  # (1, S, V)
    offset = 0
    for d in docs_meta:
        L = d["L"]
        with torch.no_grad():
            ref = _run_unpacked(tiny_llama, d["input_ids"])
        torch.testing.assert_close(
            got[:, offset:offset + L], ref, rtol=1e-4, atol=1e-5,
        )
        offset += L


# ─── eval-side collator ───────────────────────────────────────────────────────


def test_pad_to_max_collator_static_shape():
    seq = 12
    pad_id = 0
    features = [
        {"input_ids": [10, 11, 12, 13, 14], "prompt_len": 2},
        {"input_ids": [20, 21], "prompt_len": 1},
    ]
    coll = PadToMaxCollator(seq_length=seq, pad_token_id=pad_id)
    batch = coll(features)
    assert batch["input_ids"].shape == (2, seq)
    assert batch["labels"].shape == (2, seq)
    assert batch["attention_mask"].shape == (2, seq)
    # Row 0: first 5 are real, rest pad.
    assert batch["input_ids"][0, :5].tolist() == [10, 11, 12, 13, 14]
    assert batch["input_ids"][0, 5:].tolist() == [pad_id] * (seq - 5)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1] + [0] * (seq - 5)
    # Labels: -100 on prompt (first 2) and on pad tail; real ids in the middle.
    assert batch["labels"][0, :2].tolist() == [-100, -100]
    assert batch["labels"][0, 2:5].tolist() == [12, 13, 14]
    assert batch["labels"][0, 5:].tolist() == [-100] * (seq - 5)
    # Row 1: first 2 real, rest pad; entirely prompt-masked since prompt_len=1
    # and after slicing only ids[1] is response.
    assert batch["input_ids"][1, :2].tolist() == [20, 21]
    assert batch["labels"][1, :1].tolist() == [-100]
    assert batch["labels"][1, 1:2].tolist() == [21]
    assert batch["labels"][1, 2:].tolist() == [-100] * (seq - 2)


def test_pad_to_max_eval_loss_equivalence(tiny_llama):
    """Eval loss under PadToMaxCollator should match per-doc unpadded
    forward. Pad positions must not contribute to logits/loss."""
    import torch.nn.functional as F

    seq = 16
    ids_a = [3, 4, 5, 6, 7, 8]
    ids_b = [11, 12, 13]
    features = [
        {"input_ids": ids_a, "prompt_len": 0},
        {"input_ids": ids_b, "prompt_len": 0},
    ]
    coll = PadToMaxCollator(seq_length=seq, pad_token_id=0)
    batch = coll(features)
    out = tiny_llama(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    padded_loss = out.loss

    # Reference: per-doc CE summed over real-token predictions, normalized
    # by total kept tokens (matches HF's mean-over-non-100-tokens).
    total_loss = 0.0
    total_count = 0
    for ids in (ids_a, ids_b):
        with torch.no_grad():
            log = _run_unpacked(tiny_llama, ids)  # (1, L, V)
        shift_logits = log[:, :-1, :].contiguous()
        shift_labels = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss = total_loss + float(loss)
        total_count += shift_labels.numel()
    ref_loss = torch.tensor(total_loss / total_count)
    torch.testing.assert_close(padded_loss, ref_loss, rtol=1e-3, atol=1e-4)
