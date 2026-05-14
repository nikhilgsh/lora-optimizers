from types import SimpleNamespace

from datasets import Dataset

from lora_playground.train import tokenize_splits_with_boundary


class WhitespaceTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def test_packed_v1_tokenization_drops_prompt_only_truncations():
    rows = [
        {"prompt": "p0 p1 p2", "completion": "r0 r1"},
        {"prompt": "p0", "completion": "r0 r1"},
    ]
    train = Dataset.from_list(rows)
    eval_ds = Dataset.from_list(rows)
    args = SimpleNamespace(max_seq_length=3, tokenize_num_proc=None)

    train_tok, eval_tok = tokenize_splits_with_boundary(
        train, eval_ds, WhitespaceTokenizer(), args,
    )

    assert len(train_tok) == 1
    assert len(eval_tok) == 1
    assert train_tok[0]["prompt_len"] == 1
    assert train_tok[0]["prompt_len"] < len(train_tok[0]["input_ids"])
