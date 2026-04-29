# Model and Dataset Selection

## Criteria

The default course should measure code adaptation, not polish an already code-specialized model. Prefer:

- general pretrained base models, not instruct or coder variants;
- permissive, non-gated weights when possible;
- single-GPU LoRA feasibility;
- code-instruction data with clean prompt/response fields;
- moderate download size for iteration.

## Base Model Choice

Primary: `allenai/OLMo-2-0425-1B`

Rationale: OLMo 2 1B is a modern general LM, Apache-2.0, and part of an open-science model family with released training details. At 1B parameters, it is large enough to make LoRA optimizer behavior meaningful while still fitting single-GPU iteration.

Alternatives considered:

- `Qwen/Qwen3-0.6B-Base`: very strong and Apache-2.0, but its model card explicitly includes coding, STEM, and reasoning in the pretraining mix. Good secondary course, less clean for measuring adaptation from a non-code base.
- `google/gemma-3-270m`: modern and small, but gated by Gemma license acceptance and much smaller than the intended main course.
- `HuggingFaceTB/SmolLM2-360M`: small and Apache-2.0, but explicitly trained with The Stack in the data mix.
- `EleutherAI/pythia-410m-deduped`: controlled and useful for regression tests, but older and weaker than the desired main course.

## Dataset Choice

Primary: `ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response`

Rationale: moderate size, MIT license, direct `instruction` and `response` fields, and OSS-derived code tasks. It is cleaner for a fixed LoRA adaptation course than datasets with large few-shot prompt templates or very large synthetic dumps.

Alternatives considered:

- `bigcode/self-oss-instruct-sc2-exec-filter-50k`: execution-filtered and useful as a secondary course, but has ODC-BY licensing and more BigCode/StarCoder-specific provenance.
- `ise-uiuc/Magicoder-Evol-Instruct-110K`: Apache-2.0 and decontaminated, useful as a later robustness course.
- `HuggingFaceH4/CodeAlpaca_20K`: small and convenient, best kept for quick debugging.
- `glaiveai/glaive-code-assistant-v3`: large and Apache-2.0, but too large and synthetic-heavy for the default iteration loop.

## Source Pages

- OLMo 2 1B: https://huggingface.co/allenai/OLMo-2-0425-1B
- Qwen3 0.6B Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Gemma 3 270M: https://huggingface.co/google/gemma-3-270m
- SmolLM2 360M: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- Pythia 410M deduped: https://huggingface.co/EleutherAI/pythia-410m-deduped
- Magicoder OSS Instruct: https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response
- BigCode self-OSS instruct: https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k
- CodeAlpaca 20K: https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K
- Glaive code assistant v3: https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3
