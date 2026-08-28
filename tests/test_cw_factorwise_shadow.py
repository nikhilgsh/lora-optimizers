"""Read-only current-gradient/identity shadows for factorwise preconditioning."""

import copy
import json
import math
import types

import pytest
import torch
import torch.nn as nn

from lora_playground.optim import (
    CurvatureWhitenLoRA,
    _effective_step_shadow_stats,
    _small_slot_microbatch_moments,
    _skinny_quadratic_stats,
    _skinny_tangent_fro_inner,
    _skinny_tangent_fro_norm,
    _skinny_tangent_singular_values,
)


class _LoraPair(nn.Module):
    def __init__(self, d_in=9, d_out=7, r=4):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        nn.init.normal_(self.lora_A["default"].weight, std=0.2)
        nn.init.normal_(self.lora_B["default"].weight, std=0.2)


class _MultiShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = _LoraPair(d_in=9, d_out=7, r=4)
        self.p1 = _LoraPair(d_in=11, d_out=7, r=4)
        self.p2 = _LoraPair(d_in=9, d_out=8, r=4)


def _make_model(seed=0):
    torch.manual_seed(seed)
    return _LoraPair()


def _make_opt(model, *, heavy, diagnostics_every=1, **overrides):
    kwargs = dict(
        lr=3e-2,
        curvature_beta=0.5,
        delta=1e-4,
        ns_steps=8,
        higham_iters=8,
        use_polar=True,
        polar_method="polar_express",
        precond_method="gram_ns",
        kl_coupled=True,
        soap_v=False,
        cw_picard_iters=1,
        cw_nesterov=True,
        precond="factorwise",
        msign="full",
        log_basic_diagnostics=False,
        log_heavy_diagnostics=heavy,
        diagnostics_every=diagnostics_every,
    )
    kwargs.update(overrides)
    return CurvatureWhitenLoRA(model, **kwargs)


def _assign_gradients(opt, generator):
    for A, B in opt.pairs:
        A.grad = torch.randn(A.shape, generator=generator)
        B.grad = torch.randn(B.shape, generator=generator)


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if torch.is_tensor(a):
            assert torch.equal(a, b), key
        else:
            assert a == b, key


def _shadow_events(text):
    return [json.loads(line) for line in text.splitlines()
            if line.startswith("{") and json.loads(line).get("event") == "cw_shadow_step"]


def test_skinny_tangent_and_quadratic_formulas_match_dense():
    torch.manual_seed(4)
    B = torch.randn(2, 6, 3)
    A = torch.randn(2, 3, 5)
    dA_x = torch.randn_like(A)
    dB_x = torch.randn_like(B)
    dA_y = torch.randn_like(A)
    dB_y = torch.randn_like(B)

    tx = B @ dA_x + dB_x @ A
    ty = B @ dA_y + dB_y @ A
    want_inner = (tx * ty).sum(dim=(-2, -1))
    want_norm = tx.flatten(1).norm(dim=1)
    assert torch.allclose(
        _skinny_tangent_fro_inner(B, A, dA_x, dB_x, dA_y, dB_y),
        want_inner, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _skinny_tangent_fro_norm(B, A, dA_x, dB_x),
        want_norm, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _skinny_tangent_singular_values(B, A, dA_x, dB_x),
        torch.linalg.svdvals(tx), atol=1e-5, rtol=1e-5)

    # Rank-one quadratic products make the PI target exact and stable-rank one.
    u = torch.randn(2, 6, 1)
    v = torch.randn(2, 1, 5)
    dB = torch.cat((u, torch.zeros(2, 6, 2)), dim=2)
    dA = torch.cat((v, torch.zeros(2, 2, 5)), dim=1)
    dense = dB @ dA
    fro, sigma, stable_rank, guard_hit = _skinny_quadratic_stats(dA, dB)
    assert torch.allclose(fro, dense.flatten(1).norm(dim=1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(sigma, fro, atol=1e-5, rtol=1e-5)
    assert torch.allclose(stable_rank, torch.ones_like(stable_rank), atol=1e-5, rtol=1e-5)
    assert not bool(guard_hit.any())


def test_quadratic_block_pi_catches_top_direction_orthogonal_to_ones():
    a = 2.0 ** -0.5
    U = torch.tensor([
        [a, a, 0.0, 0.0],
        [-a, a, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    dB = (U @ torch.diag(torch.tensor([10.0, 1.0, 1.0, 1.0]))).unsqueeze(0)
    dA = torch.eye(4).unsqueeze(0)
    _, sigma, stable_rank, guard_hit = _skinny_quadratic_stats(dA, dB)
    assert torch.allclose(sigma, torch.tensor([10.0]), atol=1e-4, rtol=1e-4)
    assert torch.allclose(stable_rank, torch.tensor([1.03]), atol=1e-4, rtol=1e-4)
    assert not bool(guard_hit.any())


def test_quadratic_block_pi_matches_exact_r16_reference():
    torch.manual_seed(19)
    dB = torch.randn(8, 48, 16)
    dA = torch.randn(8, 16, 64)
    fro, sigma, stable_rank, guard_hit = _skinny_quadratic_stats(dA, dB)
    exact_sigma = torch.linalg.svdvals(dB @ dA)[:, 0]
    exact_stable_rank = fro.square() / exact_sigma.square()
    assert torch.allclose(sigma, exact_sigma, atol=1e-5, rtol=1e-3)
    assert torch.allclose(stable_rank, exact_stable_rank, atol=1e-5, rtol=2e-3)
    assert not bool(guard_hit.any())


def test_small_slot_uncentered_is_centered_plus_mean_outer_with_zero_slot():
    torch.manual_seed(67)
    # Unequal microbatch gradients plus an explicit skipped zero-token slot.
    gA = torch.stack((
        torch.randn(2, 4, 7),
        torch.zeros(2, 4, 7),
        3.0 * torch.randn(2, 4, 7),
    ))
    gB = torch.stack((
        torch.randn(2, 6, 4),
        torch.zeros(2, 6, 4),
        0.25 * torch.randn(2, 6, 4),
    ))
    Q_inv = torch.rand(2, 7) + 0.2
    P_inv = torch.rand(2, 6) + 0.2
    moments = _small_slot_microbatch_moments(gA, gB, Q_inv, P_inv)

    assert torch.allclose(
        moments["P_A_uncentered"],
        moments["P_A_centered"] + moments["P_A_mean_outer"],
        atol=2e-6, rtol=2e-6,
    )
    assert torch.allclose(
        moments["Q_B_uncentered"],
        moments["Q_B_centered"] + moments["Q_B_mean_outer"],
        atol=2e-6, rtol=2e-6,
    )
    for side in ("P_A", "Q_B"):
        assert torch.all(
            moments[f"small_slot_{side}_moment_identity_rel_residual"] < 2e-6)
        share = moments[f"small_slot_{side}_mean_outer_trace_share"]
        assert torch.all((share >= 0.0) & (share <= 1.0 + 2e-6))


def test_bf16_effective_shadow_matches_explicit_production_cast_add():
    A_native = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    B_native = torch.full((1, 4, 2), 0.5, dtype=torch.bfloat16)
    A = A_native.float()
    B = B_native.float()
    dA = torch.tensor([[
        [1e-4, 2e-2, -3e-2],
        [4e-2, -1e-4, 5e-2],
    ]])
    dB = torch.tensor([[
        [1e-4, 2e-2],
        [-2e-2, 3e-2],
        [4e-2, -1e-4],
        [-5e-2, 6e-2],
    ]])
    heldout_gA = torch.randn_like(dA)
    heldout_gB = torch.randn_like(dB)

    stats, effective = _effective_step_shadow_stats(
        A, B, {"probe": (dA, dB)}, torch.bfloat16, torch.bfloat16,
        heldout_gA=heldout_gA, heldout_gB=heldout_gB)

    A_post = A_native.clone()
    B_post = B_native.clone()
    A_post.add_(dA.to(torch.bfloat16))
    B_post.add_(dB.to(torch.bfloat16))
    dA_ref = A_post.float() - A_native.float()
    dB_ref = B_post.float() - B_native.float()
    assert torch.equal(effective["probe"][0], dA_ref)
    assert torch.equal(effective["probe"][1], dB_ref)
    assert dA_ref[0, 0, 0] == 0
    assert dA_ref[0, 1, 1] == 0
    assert dB_ref[0, 0, 0] == 0
    assert dB_ref[0, 2, 1] == 0

    ideal_factor = torch.cat((dA.flatten(1), dB.flatten(1)), dim=1)
    effective_factor = torch.cat((dA_ref.flatten(1), dB_ref.flatten(1)), dim=1)
    factor_cos = torch.nn.functional.cosine_similarity(
        effective_factor, ideal_factor, dim=1)
    factor_ratio = effective_factor.norm(dim=1) / ideal_factor.norm(dim=1)
    ideal_tangent = B @ dA + dB @ A
    effective_tangent = B @ dA_ref + dB_ref @ A
    tangent_cos = torch.nn.functional.cosine_similarity(
        effective_tangent.flatten(1), ideal_tangent.flatten(1), dim=1)
    tangent_ratio = (
        effective_tangent.flatten(1).norm(dim=1)
        / ideal_tangent.flatten(1).norm(dim=1))
    heldout_effective = (
        (heldout_gA * dA_ref).sum(dim=(-2, -1))
        + (heldout_gB * dB_ref).sum(dim=(-2, -1)))

    assert torch.equal(
        stats["shadow_effective_changed_frac_A_probe"],
        (dA_ref != 0).float().mean(dim=(-2, -1)))
    assert torch.equal(
        stats["shadow_effective_changed_frac_B_probe"],
        (dB_ref != 0).float().mean(dim=(-2, -1)))
    assert torch.allclose(stats["shadow_effective_factor_cos_probe"], factor_cos)
    assert torch.allclose(
        stats["shadow_effective_factor_norm_ratio_probe"], factor_ratio)
    assert torch.allclose(stats["shadow_effective_tangent_cos_probe"], tangent_cos)
    assert torch.allclose(
        stats["shadow_effective_tangent_norm_ratio_probe"], tangent_ratio)
    assert torch.equal(
        stats["shadow_heldout_predicted_loss_change_effective_probe"],
        heldout_effective)


def test_shadow_is_bit_exactly_non_mutating(capsys):
    model_plain = _make_model(seed=3)
    model_shadow = copy.deepcopy(model_plain)
    plain = _make_opt(model_plain, heavy=False)
    shadow = _make_opt(model_shadow, heavy=True)
    grad_gen = torch.Generator().manual_seed(11)

    for _ in range(3):
        grads = []
        for A, B in plain.pairs:
            grads.append((torch.randn(A.shape, generator=grad_gen),
                          torch.randn(B.shape, generator=grad_gen)))
        for opt in (plain, shadow):
            for (A, B), (gA, gB) in zip(opt.pairs, grads):
                A.grad, B.grad = gA.clone(), gB.clone()
            if opt is shadow:
                rng_before = torch.random.get_rng_state().clone()
                opt.step()
                assert torch.equal(torch.random.get_rng_state(), rng_before)
            else:
                opt.step()

    assert _shadow_events(capsys.readouterr().out)
    for (A0, B0), (A1, B1) in zip(plain.pairs, shadow.pairs):
        assert torch.equal(A0, A1)
        assert torch.equal(B0, B1)
    assert plain.pair_state.keys() == shadow.pair_state.keys()
    for i in plain.pair_state:
        _assert_state_equal(plain.pair_state[i], shadow.pair_state[i])


def test_step1_identity_null_makes_all_shadow_directions_coincide(capsys):
    model = _make_model(seed=5)
    opt = _make_opt(model, heavy=True)
    _assign_gradients(opt, torch.Generator().manual_seed(17))
    opt.step()
    event, = _shadow_events(capsys.readouterr().out)
    for field in (
        "shadow_factor_cos_fresh_lagged_median",
        "shadow_factor_cos_identity_lagged_median",
        "shadow_factor_cos_fresh_identity_median",
        "shadow_tangent_cos_fresh_lagged_median",
        "shadow_tangent_cos_identity_lagged_median",
        "shadow_tangent_cos_fresh_identity_median",
    ):
        assert abs(event[field] - 1.0) < 2e-6, (field, event[field])
    assert event["shadow_tangent_stable_rank_identity_median"] == pytest.approx(
        event["shadow_tangent_stable_rank_lagged_median"], rel=1e-5)


def test_heldout_gradient_emits_global_predicted_loss_changes(capsys):
    model = _make_model(seed=31)
    opt = _make_opt(model, heavy=True)
    _assign_gradients(opt, torch.Generator().manual_seed(53))
    hold_gen = torch.Generator().manual_seed(59)
    opt._heldout_factor_grads = [
        (torch.randn(A.shape, generator=hold_gen),
         torch.randn(B.shape, generator=hold_gen))
        for A, B in opt.pairs
    ]
    opt.step()
    event, = _shadow_events(capsys.readouterr().out)
    for label in ("lagged", "fresh", "identity"):
        key = f"shadow_heldout_predicted_loss_change_{label}"
        assert math.isfinite(event[key + "_sum"])
        assert event[key + "_sum"] == pytest.approx(event[key + "_median"])
        weak = f"shadow_heldout_predicted_loss_change_weak_{label}"
        strong = f"shadow_heldout_predicted_loss_change_strong_{label}"
        assert event[weak + "_sum"] + event[strong + "_sum"] == pytest.approx(
            event[key + "_sum"], abs=2e-6)
        for side in ("A", "B"):
            frac = event[f"shadow_update_weak_frac_{side}_{label}_median"]
            assert 0.0 <= frac <= 1.0
    for side in ("A", "B"):
        assert 0.0 <= event[f"shadow_factor_energy_weak_frac_{side}_median"] <= 1.0
        assert 0.0 <= event[f"shadow_train_grad_weak_frac_{side}_median"] <= 1.0
        assert 0.0 <= event[f"shadow_heldout_grad_weak_frac_{side}_median"] <= 1.0


def test_anisotropic_current_gradient_shadow_separates_from_lagged(capsys):
    model = _make_model(seed=7)
    opt = _make_opt(model, heavy=True, curvature_beta=0.0)
    st = opt.pair_state[0]
    r = st["P_A"].shape[0]
    st["step"] = 1
    st["P_A"].copy_(torch.diag(torch.tensor([1.0, 0.2, 0.04, 0.008])))
    st["Q_B"].copy_(torch.diag(torch.tensor([0.01, 0.05, 0.25, 1.0])))
    st["Q"].fill_(1.0)
    st["P"].fill_(1.0)
    _assign_gradients(opt, torch.Generator().manual_seed(23))
    opt.step()
    event, = _shadow_events(capsys.readouterr().out)
    assert event["shadow_PA_consumed_fresh_rel_change_median"] > 0.1
    assert event["shadow_QB_consumed_fresh_rel_change_median"] > 0.1
    assert event["shadow_tangent_cos_fresh_lagged_median"] < 0.999


def test_shadow_fresh_recurrence_uses_beta_old_d_and_dimension_normalizers(monkeypatch, capsys):
    import lora_playground.optim as optim_mod
    from lora_playground.spectral import lambda_max_power_iter_psd_batched

    model = _make_model(seed=21)
    opt = _make_opt(model, heavy=True, curvature_beta=0.6)
    st = opt.pair_state[0]
    st["step"] = 1
    st["P_A"].copy_(torch.tensor([
        [2.0, 0.1, 0.0, 0.0],
        [0.1, 0.7, 0.1, 0.0],
        [0.0, 0.1, 0.3, 0.05],
        [0.0, 0.0, 0.05, 0.1],
    ]))
    st["Q_B"].copy_(torch.tensor([
        [0.2, 0.02, 0.0, 0.0],
        [0.02, 0.4, 0.03, 0.0],
        [0.0, 0.03, 0.8, 0.04],
        [0.0, 0.0, 0.04, 1.6],
    ]))
    st["Q"].copy_(torch.linspace(0.2, 1.4, st["Q"].numel()))
    st["P"].copy_(torch.linspace(1.5, 0.3, st["P"].numel()))
    _assign_gradients(opt, torch.Generator().manual_seed(41))
    A, B = opt.pairs[0]
    gA, gB = A.grad.float().clone(), B.grad.float().clone()
    PA0, QB0 = st["P_A"].clone(), st["Q_B"].clone()
    Q0, P0 = st["Q"].clone(), st["P"].clone()
    Q_isqrt = opt._rdinv(Q0.unsqueeze(0), partner_trace=P0.sum().reshape(1, 1))[0]
    P_isqrt = opt._rdinv(P0.unsqueeze(0), partner_trace=Q0.sum().reshape(1, 1))[0]
    cb = opt.curvature_beta
    PA_raw = cb * PA0 + ((1.0 - cb) / gA.shape[1]) * (
        (gA * Q_isqrt.square().unsqueeze(0)) @ gA.T)
    QB_raw = cb * QB0 + ((1.0 - cb) / gB.shape[0]) * (
        gB.T @ (gB * P_isqrt.square().unsqueeze(1)))
    lam_PA, _ = lambda_max_power_iter_psd_batched(PA_raw.unsqueeze(0), n_iters=8)
    lam_QB, _ = lambda_max_power_iter_psd_batched(QB_raw.unsqueeze(0), n_iters=8)
    want_PA = PA_raw / lam_PA[0]
    want_QB = QB_raw / lam_QB[0]

    inverse_root_inputs = []
    real_invroot = optim_mod.gram_ns_inv_sqrt

    def recording_invroot(S, *args, **kwargs):
        inverse_root_inputs.append(S.detach().clone())
        return real_invroot(S, *args, **kwargs)

    direction_inputs = []
    real_direction = opt._cw_shadow_direction

    def recording_direction(self, **kwargs):
        direction_inputs.append((kwargs["PAh"].detach().clone(),
                                 kwargs["QBh"].detach().clone()))
        return real_direction(**kwargs)

    monkeypatch.setattr(optim_mod, "gram_ns_inv_sqrt", recording_invroot)
    opt._cw_shadow_direction = types.MethodType(recording_direction, opt)
    opt.step()
    assert len(inverse_root_inputs) == 4
    assert torch.allclose(inverse_root_inputs[2][0], want_PA, atol=1e-6, rtol=1e-5)
    assert torch.allclose(inverse_root_inputs[3][0], want_QB, atol=1e-6, rtol=1e-5)
    assert len(direction_inputs) == 2
    eye = torch.eye(PA0.shape[0])
    assert torch.equal(direction_inputs[1][0][0], eye)
    assert torch.equal(direction_inputs[1][1][0], eye)
    assert len(_shadow_events(capsys.readouterr().out)) == 1


def test_anisotropic_warm_cache_lagged_shadow_replays_actual_update(capsys):
    model = _make_model(seed=25)
    opt = _make_opt(model, heavy=False)
    _assign_gradients(opt, torch.Generator().manual_seed(43))
    opt.step()  # populate all production sigma caches
    st = opt.pair_state[0]
    st["P_A"].copy_(torch.diag(torch.tensor([1.0, 0.3, 0.08, 0.02])))
    st["Q_B"].copy_(torch.diag(torch.tensor([0.03, 0.12, 0.4, 1.0])))
    st["Q"].copy_(torch.linspace(0.4, 1.3, st["Q"].numel()))
    st["P"].copy_(torch.linspace(1.2, 0.35, st["P"].numel()))
    opt.log_heavy_diagnostics = True

    real_shadow = opt._cw_shadow_factorwise_records
    replay_checked = []

    def checking_shadow(self, **kwargs):
        replay_A, replay_B = self._cw_shadow_direction(
            PAh=kwargs["PAh_lag"],
            QBh=kwargs["QBh_lag"],
            mhatA=kwargs["mhatA"],
            mhatB=kwargs["mhatB"],
            Q_isqrt=kwargs["Q_isqrt"],
            P_isqrt=kwargs["P_isqrt"],
            cA=kwargs["cA"],
            cB=kwargs["cB"],
            rho=kwargs["rho"],
            v_sigma_WA=kwargs["v_sigma_WA"],
            v_sigma_WB=kwargs["v_sigma_WB"],
        )
        assert torch.equal(replay_A, kwargs["dA_lag"])
        assert torch.equal(replay_B, kwargs["dB_lag"])
        replay_checked.append(True)
        return real_shadow(**kwargs)

    opt._cw_shadow_factorwise_records = types.MethodType(checking_shadow, opt)
    _assign_gradients(opt, torch.Generator().manual_seed(47))
    opt.step()
    assert replay_checked == [True]
    assert len(_shadow_events(capsys.readouterr().out)) == 1


def test_unsupported_factorwise_shadow_fails_before_mutation():
    model = _make_model(seed=9)
    opt = _make_opt(model, heavy=True, use_polar=False)
    before_params = [p.detach().clone() for p in model.parameters()]
    before_state = copy.deepcopy(opt.pair_state)
    _assign_gradients(opt, torch.Generator().manual_seed(29))
    with pytest.raises(ValueError, match="fixed production-path shadow"):
        opt.step()
    for p, before in zip(model.parameters(), before_params):
        assert torch.equal(p, before)
    for i in before_state:
        _assert_state_equal(opt.pair_state[i], before_state[i])


def test_shadow_emits_only_on_requested_cadence(capsys):
    model = _make_model(seed=13)
    opt = _make_opt(model, heavy=True, diagnostics_every=2)
    _assign_gradients(opt, torch.Generator().manual_seed(31))
    opt.step()
    assert not _shadow_events(capsys.readouterr().out)
    _assign_gradients(opt, torch.Generator().manual_seed(37))
    opt.step()
    events = _shadow_events(capsys.readouterr().out)
    assert len(events) == 1
    assert events[0]["step"] == 2


def test_shadow_aggregates_all_shape_groups_in_one_finite_event(capsys):
    torch.manual_seed(33)
    model = _MultiShape()
    opt = _make_opt(model, heavy=True)
    _assign_gradients(opt, torch.Generator().manual_seed(53))
    opt.step()
    event, = _shadow_events(capsys.readouterr().out)
    assert event["n_pairs"] == 3
    assert event["fresh_semantics"].startswith("P_A/Q_B include g_t")
    assert "not a one-sided trajectory" in event["identity_semantics"]
    for key, value in event.items():
        if key.startswith("shadow_"):
            assert torch.isfinite(torch.tensor(value)), (key, value)
