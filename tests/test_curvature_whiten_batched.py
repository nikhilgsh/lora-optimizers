"""CurvatureWhitenLoRA's batched step must give a pair the same trajectory
whichever shape group it lands in.

`_cw_apply_grouped` buckets pairs by `(d_in, d_out)` and runs each bucket through
one stacked `torch.stack` / bmm / batched-Newton-Schulz sequence. The risk that
creates is a REDUCTION ACROSS THE BATCH AXIS that should have stayed per-item — a
`sum()` missing its `dim=`, a `sigma_max` estimated over the stack instead of per
matrix, a normalizer divided by the wrong shape. Such a bug makes a pair's update
depend on which OTHER pairs happen to share its shape.

So each test here runs one pair TWICE at identical gradients: once alone (a
singleton group) and once alongside a shape-identical companion (a group of 2).
The two trajectories must agree. That property is what batching is supposed to
guarantee, and nothing outside `_cw_apply_grouped` has to exist for it to be
checkable.

This file previously compared `_batched_step=True` against `_batched_step=False`,
i.e. against `_cw_apply_per_pair` — a second 219-line implementation of the same
step. That comparison was retired with the implementation, for a reason that had
already bitten: the per-pair path read neither `precond_method` nor `cw_unpinned`
nor LORA_MULTIMOMENT_RESCALE, so under the production `gram_ns` inverse-sqrt it
silently applied only the DIAGONAL of the r x r curvature. Every test in this
file left `precond_method` at its `eigh` default, so the "oracle" had never once
validated a production configuration. A comparison against a second copy also
cannot catch a bug both copies share, which is what a companion-independence
check does catch.

CPU, tiny, deterministic.
"""


def test_companion_independent_polar(companion_independent):
    companion_independent(use_polar=True)


def test_companion_independent_no_polar(companion_independent):
    companion_independent(use_polar=False)


def test_companion_independent_no_radius(companion_independent):
    # --cw_no_radius (rho=lr): the magnitude rule is removed, so the sigma_max
    # estimates that remain are the ones most likely to leak across the stack.
    companion_independent(use_polar=True, soap_v=False, cw_nesterov=True,
                           cw_no_radius=True)


def test_companion_independent_no_diag_curv(companion_independent):
    # --cw_no_diag_curv (P=Q=I so C_A=B^T B): requires diag_metric.
    companion_independent(use_polar=True, soap_v=False, cw_nesterov=True,
                           diag_metric=True, cw_no_diag_curv=True)


def test_companion_independent_diag_shampoo(companion_independent):
    # The diag-shampoo arm: diag_metric=True, soap_v=False, kl_coupled=False --
    # the branch that recomputes P_A/Q_B from the factors each step.
    companion_independent(use_polar=True, diag_metric=True, soap_v=False,
                           kl_coupled=False)


def test_companion_independent_flat_outer(companion_independent):
    # The kl-diag-polar-flatout arm: flat_outer=True skips the un-whiten.
    companion_independent(use_polar=True, diag_metric=True, soap_v=False,
                           kl_coupled=True, flat_outer=True)


def test_companion_independent_solved_rho(companion_independent):
    # cw_solved_rho: the post-Picard solved-magnitude rescale runs a
    # sum-of-products power iteration over the stack, so it is exactly the shape
    # of code this file exists to check.
    companion_independent(use_polar=True, diag_metric=True, soap_v=False,
                           kl_coupled=True, cw_nesterov=True, cw_solved_rho=True)


def test_companion_independent_at_the_production_precond_method(companion_independent):
    """The configuration the retired per-pair oracle never validated.

    Every test in this file used to leave `precond_method` at its "eigh"
    default while comparing against `_cw_apply_per_pair`, which ignored the
    field entirely -- so `gram_ns`, the inverse-sqrt every production sweep
    uses, was the one path with no equivalence coverage at all.
    """
    companion_independent(use_polar=True, diag_metric=True, soap_v=False,
                           kl_coupled=True, cw_nesterov=True,
                           precond_method="gram_ns", higham_iters=8,
                           polar_method="polar_express", ns_steps=8)
