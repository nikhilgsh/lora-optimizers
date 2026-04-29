import sys
from pathlib import Path

import torch
import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.utils import collect_lora_pairs, solve_spd, solve_sylvester, spdify


# --------------------- helpers ---------------------

def rand_spd(r, eps=1e-5, dtype=torch.float32, device="cpu"):
    M = torch.randn(r, r, dtype=dtype, device=device)
    return spdify(M.T @ M, eps)


# --------------------- tests -----------------------


@pytest.mark.parametrize("r", [1, 4, 16])
def test_spdify_symmetry_pd(r):
    torch.manual_seed(0)
    base = rand_spd(r, eps=1e-3, dtype=torch.float32)
    perturb = 1e-2 * torch.randn(r, r, dtype=torch.float32)
    M = base + perturb
    A = spdify(M, eps=1e-3)
    assert A.dtype == torch.float32
    assert torch.allclose(A, A.T, atol=1e-5)
    eig = torch.linalg.eigvalsh(A)
    assert torch.all(eig > 0)


@pytest.mark.parametrize("r,nrhs", [(3, 1), (5, 2), (8, 4)])
def test_solve_spd_linear_system(r, nrhs):
    torch.manual_seed(1)
    A = rand_spd(r, eps=1e-6, dtype=torch.float32)
    X_true = torch.randn(r, nrhs, dtype=A.dtype)
    B = A @ X_true
    X = solve_spd(A, B)
    assert torch.allclose(X, X_true, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("r", [2, 5, 12])
def test_solve_sylvester_accuracy(r):
    torch.manual_seed(2)
    SB = rand_spd(r, eps=1e-6, dtype=torch.float32)
    SA = rand_spd(r, eps=1e-6, dtype=torch.float32)
    K_true = torch.randn(r, r, dtype=torch.float32)
    RHS = K_true @ SA + SB @ K_true
    K = solve_sylvester(SB, SA, RHS)
    resid = K @ SA + SB @ K - RHS
    # Sylvester residual should be tiny
    assert torch.linalg.norm(resid) < 5e-4


def test_collect_lora_pairs_single_adapter():
    torch.manual_seed(3)
    r, in_f, out_f = 4, 7, 9

    class Weight(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(*shape))

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            # mimic PEFT single-adapter attributes
            self.lora_A = Weight((r, in_f))
            self.lora_B = Weight((out_f, r))

    model = nn.Sequential(nn.Linear(5, 5), Dummy(), nn.ReLU())
    pairs = collect_lora_pairs(model)
    assert len(pairs) == 1
    A, B = pairs[0]
    assert A.shape == (r, in_f)
    assert B.shape == (out_f, r)


def test_collect_lora_pairs_multi_adapter():
    torch.manual_seed(4)
    r, in_f, out_f = 3, 6, 8

    class Weight(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(*shape))

    class DummyMulti(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict(
                {
                    "default": Weight((r, in_f)),
                    "other": Weight((r, in_f)),
                }
            )
            self.lora_B = nn.ModuleDict(
                {
                    "default": Weight((out_f, r)),
                    "other": Weight((out_f, r)),
                }
            )

    model = DummyMulti()
    # all adapters
    all_pairs = collect_lora_pairs(model)
    assert len(all_pairs) == 2
    # specific adapter
    def_pairs = collect_lora_pairs(model, adapter_name="default")
    assert len(def_pairs) == 1
    A, B = def_pairs[0]
    assert A.shape == (r, in_f) and B.shape == (out_f, r)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_utils_dtype_device_roundtrip(dtype):
    # basic smoke: utils handle dtype/device without throwing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = 6
    SB = rand_spd(r, eps=1e-6, dtype=dtype, device=device)
    SA = rand_spd(r, eps=1e-6, dtype=dtype, device=device)
    RHS = torch.randn(r, r, dtype=dtype, device=device).float()
    # Sylvester
    K = solve_sylvester(SB, SA, RHS)
    assert K.shape == (r, r)
    # SPD solve (with multi-RHS)
    B = torch.randn(r, 3, dtype=dtype, device=device).float()
    X = solve_spd(SB, B)
    assert X.shape == (r, 3)
