"""Stream-tokenize a pretrain corpus (OpenWebMath) into contiguous fixed-width
slots and save to disk as HuggingFace Arrow datasets — the *continued-pretraining*
analogue of `scripts/data/prepare_data.py` (which is for instruction tuning).

Faithful to `tanya_results/owm300m_polar_sweep.md`:
  - Model tokenizer: Qwen/Qwen3-0.6B-Base.
  - First ~1M tokens -> validation; next ~320M tokens -> train.
  - CONTIGUOUS packing: documents are concatenated (eos between docs) and sliced
    into fixed `seq_length` windows. No prompt masking (it is a corpus, not
    (prompt, response) pairs): every token is a next-token target, so
    `labels == input_ids`. No document-boundary attention isolation — each window
    is one full causal block (`doc_lens=[seq_length]`, contiguous position_ids),
    which is what the block-diagonal mask reduces to for a single-doc slot.

Output schema (matches the repo's packed_v1.1 contract so train.py reads it
directly via `--data_dir`):
  train/  rows: {input_ids, labels, position_ids, doc_lens}  (all len==seq_length)
  eval/   rows: {input_ids, prompt_len(=0)}                   (PadToMaxCollator)

Example:
    python scripts/data/prepare_openwebmath_pretrain.py \
        --model_name Qwen/Qwen3-0.6B-Base \
        --seq_length 2048 \
        --eval_windows 512 \
        --train_slots 160000 \
        --out_dir data/openwebmath_qwen3_320m_packed_seq2048

160000 train slots @ 2048 = ~327.7M tokens; with batch 4 x grad_accum 4 a
9000-step run consumes 144000 slots, leaving ~10% single-pass margin.
"""
import argparse
import os
import time

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def _window_stream(dataset_name, text_field, tokenizer, seq_length, batch_docs=1000):
    """Yield contiguous `seq_length`-wide token windows (list[int]) from a
    streaming corpus. Documents are tokenized in batches and concatenated with
    a trailing eos; full windows are drained from a rolling buffer. The final
    partial window (< seq_length) is dropped."""
    eos = tokenizer.eos_token_id
    ds = load_dataset(dataset_name, split="train", streaming=True)
    buf = []
    batch = []

    def _drain():
        while len(buf) >= seq_length:
            yield buf[:seq_length]
            del buf[:seq_length]

    for row in ds:
        batch.append(row[text_field])
        if len(batch) >= batch_docs:
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos)
            batch = []
            yield from _drain()
    if batch:
        for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
            buf.append(eos)
        yield from _drain()


def _eval_gen(dataset_name, text_field, model_name, seq_length, n_windows):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    for i, w in enumerate(_window_stream(dataset_name, text_field, tok, seq_length)):
        if i >= n_windows:
            break
        yield {"input_ids": w, "prompt_len": 0}


def _train_gen(dataset_name, text_field, model_name, seq_length, skip_windows, n_slots):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pos = list(range(seq_length))
    emitted = 0
    for i, w in enumerate(_window_stream(dataset_name, text_field, tok, seq_length)):
        if i < skip_windows:
            continue
        yield {
            "input_ids": w,
            "labels": list(w),               # all-token loss: no -100
            "position_ids": pos,
            "doc_lens": [seq_length],        # one full causal block (contiguous)
        }
        emitted += 1
        if emitted >= n_slots:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--dataset_name", default="open-web-math/open-web-math")
    p.add_argument("--text_field", default="text")
    p.add_argument("--seq_length", type=int, default=2048)
    p.add_argument("--eval_windows", type=int, default=512,
                   help="# validation windows (512 @ 2048 ~= 1.05M tokens).")
    p.add_argument("--train_slots", type=int, default=160000,
                   help="# train slots; must exceed max_steps*batch*grad_accum.")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    print(f"model={args.model_name} dataset={args.dataset_name} seq={args.seq_length}", flush=True)
    print(f"eval_windows={args.eval_windows} train_slots={args.train_slots}", flush=True)

    common = dict(
        dataset_name=args.dataset_name,
        text_field=args.text_field,
        model_name=args.model_name,
        seq_length=args.seq_length,
    )

    t0 = time.perf_counter()
    print("building eval split (first windows) ...", flush=True)
    eval_ds = Dataset.from_generator(
        _eval_gen,
        gen_kwargs={**common, "n_windows": args.eval_windows},
    )
    print(f"  eval: {len(eval_ds)} windows ({len(eval_ds)*args.seq_length:,} tokens) "
          f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    t1 = time.perf_counter()
    print("building train split (contiguous slots, skipping eval windows) ...", flush=True)
    train_ds = Dataset.from_generator(
        _train_gen,
        gen_kwargs={**common, "skip_windows": args.eval_windows, "n_slots": args.train_slots},
    )
    print(f"  train: {len(train_ds)} slots ({len(train_ds)*args.seq_length:,} tokens) "
          f"in {(time.perf_counter()-t1)/60:.1f} min", flush=True)

    train_out = os.path.join(args.out_dir, "train")
    eval_out = os.path.join(args.out_dir, "eval")
    train_ds.save_to_disk(train_out)
    eval_ds.save_to_disk(eval_out)
    print(f"saved train -> {train_out}  ({len(train_ds)} slots)", flush=True)
    print(f"saved eval  -> {eval_out}  ({len(eval_ds)} windows)", flush=True)
    print(f"total {(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
