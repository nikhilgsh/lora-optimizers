#!/usr/bin/env python3
"""Numerical checks for a polar LoRA A-factor update.

This script uses only NumPy/SciPy.  It checks:
  1. Whether the polar numerator maximizer is also the constrained optimum
     after the operator-norm scale factor is included.
  2. Whether alternating direction projection and operator-norm scaling is a
     one-round fixed point.

The stated maximization is over <Delta A, -G_A>.  With the usual polar
convention polar(M) = argmax_U <U, M>, the numerator maximizer for this
objective is polar(-H_A), not polar(H_A).  The script reports both the literal
polar(H_A) sign and the descent-signed polar(-H_A) sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize


DIN = 8
DOUT = 6
R = 3
RHO = 1.0
SEEDS = (0, 1, 2)

N_RANDOM_SCREEN = 500
N_RESTARTS = 64
MAX_ITERS = 1000


@dataclass
class OptResult:
    value: float
    U: np.ndarray
    constraint_error: float
    max_probe_improvement: float
    slsqp_successes: int
    starts_within_1e_8: int
    best_random_value: float


def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def spd_sqrt_and_invsqrt(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = linalg.eigh(M)
    if np.min(eigvals) <= 0:
        raise ValueError(f"matrix is not SPD; min eig={np.min(eigvals):.6e}")
    sqrt_M = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    invsqrt_M = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return sqrt_M, invsqrt_M


def polar(M: np.ndarray) -> np.ndarray:
    U, _, Vt = linalg.svd(M, full_matrices=False)
    return U @ Vt


def op_norm(M: np.ndarray) -> float:
    return float(linalg.svdvals(M)[0])


def left_orthonormalize(Y: np.ndarray) -> np.ndarray:
    """Return U with U U^T = I, using a QR retraction on Y^T."""
    Q, Rmat = linalg.qr(Y.T, mode="economic")

    # Deterministic QR signs make repeated runs bit-stable.
    signs = np.sign(np.diag(Rmat))
    signs[signs == 0.0] = 1.0
    Q = Q * signs
    return Q.T


def random_partial_isometry(rng: np.random.Generator) -> np.ndarray:
    return left_orthonormalize(rng.normal(size=(R, DIN)))


def delta_from_U(
    U: np.ndarray,
    C_invsqrt: np.ndarray,
    Q_invsqrt: np.ndarray,
    rho: float,
) -> np.ndarray:
    W = C_invsqrt @ U @ Q_invsqrt
    return rho * W / op_norm(W)


def objective_delta(delta_A: np.ndarray, G_A: np.ndarray) -> float:
    return float(np.sum(delta_A * (-G_A)))


def objective_from_U(
    U: np.ndarray,
    L: np.ndarray,
    C_invsqrt: np.ndarray,
    Q_invsqrt: np.ndarray,
    rho: float,
) -> float:
    numerator = float(np.sum(U * L))
    denom = op_norm(C_invsqrt @ U @ Q_invsqrt)
    return rho * numerator / denom


def objective_and_euclidean_grad(
    U: np.ndarray,
    L: np.ndarray,
    C_invsqrt: np.ndarray,
    Q_invsqrt: np.ndarray,
    rho: float,
) -> tuple[float, np.ndarray]:
    """Objective and ambient gradient for f(U)=rho*<U,L>/||C^-1/2 U Q^-1/2||_2."""
    M = C_invsqrt @ U @ Q_invsqrt
    U_svd, singular_values, Vt_svd = linalg.svd(M, full_matrices=False)
    sigma = float(singular_values[0])
    numerator = float(np.sum(U * L))
    value = rho * numerator / sigma

    # For a simple top singular value, d sigma / dM = u_1 v_1^T.
    dsigma_dM = np.outer(U_svd[:, 0], Vt_svd[0, :])
    dsigma_dU = C_invsqrt @ dsigma_dM @ Q_invsqrt
    grad = rho * (L * sigma - numerator * dsigma_dU) / (sigma * sigma)
    return value, grad


def project_to_row_stiefel_tangent(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    return G - sym(G @ U.T) @ U


def max_random_tangent_probe_improvement(
    U: np.ndarray,
    L: np.ndarray,
    C_invsqrt: np.ndarray,
    Q_invsqrt: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    n_dirs: int = 320,
) -> float:
    """Finite local check around a nonsmooth SLSQP solution.

    At the best anisotropic optima below, the denominator's top singular value
    is often tied, so a single top-singular-vector gradient is not a reliable
    stationarity certificate.  This probes random tangent directions and small
    retraction radii directly.
    """
    base = objective_from_U(U, L, C_invsqrt, Q_invsqrt, rho)
    best = base
    eps_values = (1e-5, 1e-4, 1e-3, 1e-2)
    probes_per_eps = max(1, n_dirs // len(eps_values))

    for eps in eps_values:
        for _ in range(probes_per_eps):
            Z = rng.normal(size=U.shape)
            Z = project_to_row_stiefel_tangent(U, Z)
            norm_Z = float(linalg.norm(Z, ord="fro"))
            if norm_Z == 0.0:
                continue
            Z /= norm_Z
            for sign in (-1.0, 1.0):
                trial_U = left_orthonormalize(U + sign * eps * Z)
                best = max(best, objective_from_U(trial_U, L, C_invsqrt, Q_invsqrt, rho))

    return best - base


def maximize_over_partial_isometries(
    L: np.ndarray,
    C_invsqrt: np.ndarray,
    Q_invsqrt: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    must_try: tuple[np.ndarray, ...] = (),
) -> OptResult:
    candidates: list[tuple[float, np.ndarray]] = []

    for U in must_try:
        candidates.append((objective_from_U(U, L, C_invsqrt, Q_invsqrt, rho), U))

    for _ in range(N_RANDOM_SCREEN):
        U = random_partial_isometry(rng)
        candidates.append((objective_from_U(U, L, C_invsqrt, Q_invsqrt, rho), U))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_random_value = candidates[0][0]

    starts = candidates[:N_RESTARTS]
    n_variables = R * DIN

    def unpack(x: np.ndarray) -> np.ndarray:
        return x.reshape(R, DIN)

    def objective_for_scipy(x: np.ndarray) -> float:
        return -objective_from_U(unpack(x), L, C_invsqrt, Q_invsqrt, rho)

    def gradient_for_scipy(x: np.ndarray) -> np.ndarray:
        _, grad = objective_and_euclidean_grad(unpack(x), L, C_invsqrt, Q_invsqrt, rho)
        return -grad.reshape(n_variables)

    constraints = []
    for i in range(R):
        for j in range(i, R):
            target = 1.0 if i == j else 0.0

            def constraint_value(x: np.ndarray, i: int = i, j: int = j, target: float = target) -> float:
                U = unpack(x)
                return float(U[i] @ U[j] - target)

            def constraint_jacobian(x: np.ndarray, i: int = i, j: int = j) -> np.ndarray:
                U = unpack(x)
                grad = np.zeros_like(U)
                if i == j:
                    grad[i] = 2.0 * U[i]
                else:
                    grad[i] = U[j]
                    grad[j] = U[i]
                return grad.reshape(n_variables)

            constraints.append(
                {
                    "type": "eq",
                    "fun": constraint_value,
                    "jac": constraint_jacobian,
                }
            )

    refined: list[tuple[float, np.ndarray, bool]] = []
    for _, U0 in starts:
        result = optimize.minimize(
            objective_for_scipy,
            U0.reshape(n_variables),
            jac=gradient_for_scipy,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": MAX_ITERS, "ftol": 1e-12, "disp": False},
        )
        U = unpack(result.x)
        value = objective_from_U(U, L, C_invsqrt, Q_invsqrt, rho)
        refined.append((value, U, bool(result.success)))

    refined.sort(key=lambda item: item[0], reverse=True)
    best_value, best_U, _ = refined[0]
    constraint_error = float(linalg.norm(best_U @ best_U.T - np.eye(R), ord="fro"))
    probe_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int32).max))
    max_probe_improvement = max_random_tangent_probe_improvement(
        best_U, L, C_invsqrt, Q_invsqrt, rho, probe_rng
    )
    starts_within = sum(
        abs(value - best_value) <= 1e-8 * max(1.0, abs(best_value))
        for value, _, _ in refined
    )
    slsqp_successes = sum(success for _, _, success in refined)

    return OptResult(
        value=best_value,
        U=best_U,
        constraint_error=constraint_error,
        max_probe_improvement=max_probe_improvement,
        slsqp_successes=slsqp_successes,
        starts_within_1e_8=starts_within,
        best_random_value=best_random_value,
    )


def make_case(seed: int, isotropic: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(R, DIN))
    G = rng.normal(size=(DOUT, DIN))

    if isotropic:
        B_raw = rng.normal(size=(DOUT, R))
        B, _ = linalg.qr(B_raw, mode="economic")
        P_diag = np.ones(DOUT)
        Q_diag = np.ones(DIN)
    else:
        B = rng.normal(size=(DOUT, R))
        P_diag = np.exp(rng.normal(loc=0.0, scale=1.5, size=DOUT))
        Q_diag = np.exp(rng.normal(loc=0.0, scale=1.5, size=DIN))

    P = np.diag(P_diag)
    Q = np.diag(Q_diag)
    C_A = B.T @ P @ B
    G_A = B.T @ G

    # A is included to match the setup, although W = B A is not needed for the
    # two variational/fixed-point checks.
    _ = B @ A

    return {
        "B": B,
        "G": G,
        "P": P,
        "Q": Q,
        "C_A": C_A,
        "G_A": G_A,
    }


def check_claim_1(seed: int, isotropic: bool) -> dict[str, float]:
    case = make_case(seed, isotropic=isotropic)
    C_A = case["C_A"]
    Q = case["Q"]
    G_A = case["G_A"]

    _, C_invsqrt = spd_sqrt_and_invsqrt(C_A)
    _, Q_invsqrt = spd_sqrt_and_invsqrt(Q)
    H_A = C_invsqrt @ G_A @ Q_invsqrt
    L = -H_A

    U_literal = polar(H_A)
    U_descent = polar(-H_A)

    delta_literal = delta_from_U(U_literal, C_invsqrt, Q_invsqrt, RHO)
    delta_descent = delta_from_U(U_descent, C_invsqrt, Q_invsqrt, RHO)
    obj_literal = objective_delta(delta_literal, G_A)
    obj_descent = objective_delta(delta_descent, G_A)

    if isotropic:
        exact_obj = RHO * float(np.sum(linalg.svdvals(L)))
        opt = None
    else:
        opt_rng = np.random.default_rng(100_000 + seed)
        opt = maximize_over_partial_isometries(
            L,
            C_invsqrt,
            Q_invsqrt,
            RHO,
            opt_rng,
            must_try=(U_descent, U_literal),
        )
        exact_obj = opt.value

    descent_gap = (exact_obj - obj_descent) / abs(obj_descent)
    literal_gap = (exact_obj - obj_literal) / abs(obj_literal)

    return {
        "cond_P": float(np.linalg.cond(case["P"])),
        "cond_Q": float(np.linalg.cond(case["Q"])),
        "cond_C": float(np.linalg.cond(C_A)),
        "obj_literal": obj_literal,
        "obj_descent": obj_descent,
        "obj_exact": exact_obj,
        "gap_literal": literal_gap,
        "gap_descent": descent_gap,
        "best_random": float("nan") if opt is None else opt.best_random_value,
        "constraint_error": 0.0 if opt is None else opt.constraint_error,
        "max_probe_improvement": 0.0 if opt is None else opt.max_probe_improvement,
        "slsqp_successes": 0 if opt is None else opt.slsqp_successes,
        "starts_within": 0 if opt is None else opt.starts_within_1e_8,
    }


def check_claim_2(seed: int) -> dict[str, float]:
    case = make_case(seed, isotropic=False)
    C_A = case["C_A"]
    Q = case["Q"]
    G_A = case["G_A"]

    C_sqrt, C_invsqrt = spd_sqrt_and_invsqrt(C_A)
    Q_sqrt, Q_invsqrt = spd_sqrt_and_invsqrt(Q)

    def p_dir(X: np.ndarray) -> np.ndarray:
        return C_invsqrt @ polar(C_sqrt @ X @ Q_sqrt) @ Q_invsqrt

    def p_scale(X: np.ndarray) -> np.ndarray:
        return RHO * X / op_norm(X)

    X1 = p_scale(p_dir(G_A))
    X2 = p_scale(p_dir(X1))
    X3 = p_scale(p_dir(X2))

    return {
        "norm_X1": op_norm(X1),
        "diff_21": float(linalg.norm(X2 - X1, ord="fro")),
        "diff_31": float(linalg.norm(X3 - X1, ord="fro")),
    }


def format_float(x: float) -> str:
    return f"{x:.12e}"


def main() -> None:
    print("polar_lora_update_check.py")
    print(f"dims: din={DIN}, dout={DOUT}, r={R}, rho={RHO:g}")
    print(
        "Stiefel optimizer: "
        f"{N_RANDOM_SCREEN} random screen + {N_RESTARTS} constrained SLSQP restarts"
    )
    print("Sign note: for max <Delta A, -G_A>, polar(-H_A) is the numerator maximizer.")
    print()

    anisotropic_gaps: list[float] = []
    literal_gaps: list[float] = []
    fixed_21: list[float] = []
    fixed_31: list[float] = []

    print("CLAIM 1: anisotropic P,Q")
    for seed in SEEDS:
        result = check_claim_1(seed, isotropic=False)
        anisotropic_gaps.append(result["gap_descent"])
        literal_gaps.append(result["gap_literal"])

        print(f"seed {seed}:")
        print(
            "  cond(P)="
            f"{result['cond_P']:.3f}, cond(Q)={result['cond_Q']:.3f}, "
            f"cond(C_A)={result['cond_C']:.3f}"
        )
        print(f"  obj literal polar(H_A):        {format_float(result['obj_literal'])}")
        print(f"  obj descent polar(-H_A):      {format_float(result['obj_descent'])}")
        print(f"  obj exact numerical:          {format_float(result['obj_exact'])}")
        print(f"  relative gap literal sign:    {format_float(result['gap_literal'])}")
        print(f"  relative gap descent sign:    {format_float(result['gap_descent'])}")
        print(
            "  optimizer diagnostics: "
            f"best_random={format_float(result['best_random'])}, "
            f"constraint_err={format_float(result['constraint_error'])}, "
            f"max_probe_improve={format_float(result['max_probe_improvement'])}, "
            f"slsqp_successes={int(result['slsqp_successes'])}/{N_RESTARTS}, "
            f"starts_within_1e-8={int(result['starts_within'])}/{N_RESTARTS}"
        )
    print()

    print("CLAIM 1: isotropic limit P=Q=I, B^T B=I")
    isotropic_gaps: list[float] = []
    isotropic_literal_gaps: list[float] = []
    for seed in SEEDS:
        result = check_claim_1(seed, isotropic=True)
        isotropic_gaps.append(result["gap_descent"])
        isotropic_literal_gaps.append(result["gap_literal"])
        print(f"seed {seed}:")
        print(
            "  cond(P)="
            f"{result['cond_P']:.3f}, cond(Q)={result['cond_Q']:.3f}, "
            f"cond(C_A)={result['cond_C']:.3f}"
        )
        print(f"  obj literal polar(H_A):        {format_float(result['obj_literal'])}")
        print(f"  obj descent polar(-H_A):      {format_float(result['obj_descent'])}")
        print(f"  obj exact analytic:           {format_float(result['obj_exact'])}")
        print(f"  relative gap literal sign:    {format_float(result['gap_literal'])}")
        print(f"  relative gap descent sign:    {format_float(result['gap_descent'])}")
    print()

    print("CLAIM 2: one-round fixed point for anisotropic P,Q")
    for seed in SEEDS:
        result = check_claim_2(seed)
        fixed_21.append(result["diff_21"])
        fixed_31.append(result["diff_31"])
        print(f"seed {seed}:")
        print(f"  ||X1||_2:      {format_float(result['norm_X1'])}")
        print(f"  ||X2-X1||_F:   {format_float(result['diff_21'])}")
        print(f"  ||X3-X1||_F:   {format_float(result['diff_31'])}")
    print()

    median_gap = float(np.median(anisotropic_gaps))
    max_iso_gap = float(np.max(np.abs(isotropic_gaps)))
    max_fixed_21 = float(np.max(fixed_21))
    max_fixed_31 = float(np.max(fixed_31))

    print("VERDICT")
    print(
        "(1) For the stated max <Delta A, -G_A>, descent-signed polar(-H_A) is "
        f"approximate, not exact; anisotropic relative gaps are "
        f"{[float(f'{g:.6g}') for g in anisotropic_gaps]} "
        f"(median {median_gap:.6g}). Literal polar(H_A) has the opposite sign here."
    )
    print(
        "(2) The descent-signed gap vanishes in the isotropic limit: "
        f"max |gap| = {max_iso_gap:.3e}. Literal polar(H_A) does not vanish because "
        "of the sign mismatch."
    )
    print(
        "(3) Alternation is a one-round fixed point to numerical precision: "
        f"max ||X2-X1||_F = {max_fixed_21:.3e}, "
        f"max ||X3-X1||_F = {max_fixed_31:.3e}; K>1 is a no-op."
    )


if __name__ == "__main__":
    main()
