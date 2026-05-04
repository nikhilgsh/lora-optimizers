# Candidate: orthogonal-core LoRA optimizer (UCV^T)

*2026-05-03*

## Parameterization

Replace the BA parameterization with the SVD-like factorization
$$
\Delta W = U C V^\top, \qquad U \in \mathbb{R}^{d_{\text{out}} \times r},\ V \in \mathbb{R}^{d_{\text{in}} \times r},\ C \in \mathbb{R}^{r \times r},
$$
with the constraints $U^\top U = I$ and $V^\top V = I$.

## Optimizer (single step)

Per training step, after backprop produces gradients $g_U, g_C, g_V$:

**1. Project subspace gradients to the Stiefel tangent.**
$$
g_U^\perp = (I - UU^\top) g_U, \qquad g_V^\perp = (I - VV^\top) g_V.
$$

**2. Run Adam on three separate states** $(m_U, v_U)$, $(m_C, v_C)$, $(m_V, v_V)$, with inputs $g_U^\perp, g_C, g_V^\perp$. Get bias-corrected directions $u_U, u_C, u_V$.

**3. Polar / Muon update on the subspaces.**
$$
P_U = \mathrm{polar}_{\text{NS-}j}(u_U), \qquad P_V = \mathrm{polar}_{\text{NS-}j}(u_V),
$$
$$
\mathrm{d}U = -\eta \frac{\lVert u_U \rVert_F}{\lVert P_U \rVert_F + \varepsilon} P_U, \qquad \mathrm{d}V = -\eta \frac{\lVert u_V \rVert_F}{\lVert P_V \rVert_F + \varepsilon} P_V.
$$

**4. Plain Adam step on the core.**
$$
\mathrm{d}C = -\eta \, u_C.
$$

**5. Apply with retraction on the subspaces.**
$$
U \gets \mathrm{polar}_{\text{NS-}j}(U + \mathrm{d}U), \qquad V \gets \mathrm{polar}_{\text{NS-}j}(V + \mathrm{d}V), \qquad C \gets C + \mathrm{d}C.
$$
The retraction keeps $U, V$ on the Stiefel manifold (orthonormal columns); they would drift otherwise.

No Picard loop. No $k$ hyperparameter. No core remix coefficient.

## Implementation considerations

- Custom layer module replacing PEFT's LoRA injection (current code uses PEFT's LoraConfig).
- Forward pass: $W x + (UCV^\top) x$ — three matmuls vs LoRA's two.
- Initialization: random orthogonal $U, V$ (e.g. via QR of a Gaussian matrix); zero $C$ for "no perturbation at init."
- Optimizer: three separate Adam states per layer pair; modified update logic. The existing polar / Newton–Schulz utilities can be reused for the Stiefel retraction, but need verification that they handle tall matrices ($U$ is $d_{\text{out}} \times r$ with $d_{\text{out}} \gg r$).
