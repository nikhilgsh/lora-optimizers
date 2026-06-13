# Related-work working notes

Load-bearing analysis behind the rendered Related Work section of `skeleton.tex`.
The `.tex` `%` comments point here instead of carrying the analysis inline (so the
notes are actually read).

## LoRA-Muon (`\cite{loramuon}`, concurrent)

**Structure of their paper (decoded — it is confusingly organized):**
- **Algorithm 1 = LoRA-Muon.** The method, "used throughout the paper." It is the
  partner-Gram-whitened polar with EMA momentum: $S_A=A^\top A$, $S_B=B^\top B$,
  $\Delta A = -(\eta/2)\,\mathrm{msign}(m_A S_B^{-1/2})\,S_B^{-1/2}$ (and symmetrically for $B$),
  plus a split weight decay $s=\sqrt{1-\lambda\eta}$. Their Prop 6: this equals the
  simplified LoRA-RITE core in QR coordinates → it is the iMuon/RITE shared core
  (`\Cref{prop:imuon}`). **Our both-controls-removed (double-ablation) arm reproduces exactly this.**
- **Algorithm 2 = scalar gauge rebalancing** (Appendix B.1). NOT the optimizer — an
  *optional numerical-conditioning* bolt-on: periodically rescale $(A,B)\to(A/c,cB)$ with
  $c=(\lVert A\rVert_2/\lVert B\rVert_2)^{\alpha/2}$ to equalize the factor operator norms.
  Their Fig 4 proves it is a **no-op** for LoRA-Muon (the Alg-1 update is gauge-invariant,
  so the gauge relabel cannot change the loss). $\alpha\in(0,1]$ damping; their ablation
  rebalanced every step (no specific $\alpha$ value stated in the text).
- **Appendix B.2 (Prop 7) = moment transport.** Just the correctness proof for Alg 2: under
  $(A,B)\to(cA,B/c)$ the factor gradients rescale ($g_A\to g_A/c$, $g_B\to cg_B$, because the
  product-space gradient $G=\nabla_W f$ is unchanged), so the momentum buffers must rescale
  the same way ($m_A\to m_A/c$, $m_B\to cm_B$) for the rebalance to be a clean no-op. Plumbing
  for Alg 2, not a separate idea.

**Conclusion (why we don't run Alg 2):** LoRA-Muon = Alg 1; we reproduce it. The rebalancing
(Alg 2) is optional conditioning they themselves prove is a no-op, and is about *their*
gauge-invariant method — irrelevant to our claim (our gauge-*dependent* radius is benign in
practice because the operator norms self-balance). So we run the core and note in one line
that we omit their two optional no-op add-ons.

**Other notes:**
- Scope: from-scratch TinyShakespeare ($d{=}128$, 2-layer, ~1M tok); NOT fine-tuning, NOT
  large scale (their Sec 7). Theory carries $W_\mathrm{pre}$ but experiments set it 0 — say
  "their *experiments* are from-scratch," not "their theory is pretraining-only."
- B=0 instability: Alg 1 is $B{=}0$-unstable by the same $1/\sigma_{\min}(B)$ mechanism
  (`\Cref{rem:b0}`); they never hit it (from-scratch low-rank needs nonzero factors). Untested
  at the standard $B{=}0$ adapter init (default fine-tuning init; nonzero init is a studied
  departure, Li et al. 2505.23194).
- No curvature (first-moment only); no operator-norm radius ($\eta/2$ split instead).
- "Beats dense" is narrow: rank-32 only on TinyShakespeare; rank-2 loss is *worse* than dense
  (2.156 vs 1.789) — their low-rank claim is lr-transfer, not loss.
- FLOP counts only, no wall-clock; we measure wall (`\Cref{sec:exp-walltime}`).
- Their Prop 5 (radius gauge-sensitivity) is invariance-only — no loss/stability result;
  evidenced by an adversarial ×99 rescale, no natural occurrence shown. This is the linearized
  version of our radius $\rho=\eta/(\lVert A\rVert_2+\lVert B\rVert_2)$, so it targets us — our
  rebuttal is the gauge plot (operator norms self-balance; the adversarial regime never arises).
