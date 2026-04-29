# Low-rank prototype convention

The low-rank notebook in `notebooks/low_rank.ipynb` now uses the PEFT LoRA
factor convention:

```text
A: (r, d_in)
B: (d_out, r)
W = B @ A
```

This matches PEFT's module order, where `lora_A` maps from `d_in` to `r` and
`lora_B` maps from `r` to `d_out`. The adapter weight update is therefore
`lora_B.weight @ lora_A.weight`.

The old `dual_lora` notebook used an `L @ R` convention. The mapping is:

```text
L -> B
R -> A
m -> d_out
n -> d_in
```

The scaled-gradient update in the notebook is oriented to match
`lora_playground.optim.ScaledLoRA`:

```text
S_B = B.T @ B
S_A = A @ A.T
Delta A = -lr * S_B^{-1} grad_A
Delta B = -lr * grad_B S_A^{-1}
```

The Gauss-Newton and weighted Gauss-Newton sections use the same factor
renaming, so all synthetic experiments now report and update factors in the
same order as the production PEFT optimizer code.
