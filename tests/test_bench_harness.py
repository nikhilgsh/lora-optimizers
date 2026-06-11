"""CPU contracts for the bench harness: config↔build_optimizer alignment and the
bench→registry round-trip (no GPU)."""
import torch
import torch.nn as nn

from lora_playground import timing
from lora_playground.bench import BenchConfig, protagonist_config, record_bench
from lora_playground.optim import build_optimizer


class _Pair(nn.Module):
    def __init__(self, r, di, do):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(di, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, do, bias=False)})
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A["default"].weight)
            self.lora_B["default"].weight.zero_()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([_Pair(8, 32, 32), _Pair(8, 32, 64)])


def test_protagonist_config_locked_flags():
    cfg = protagonist_config("allenai/OLMo-2-0425-1B", 256)
    assert cfg.lora_alpha == 256                  # alpha = r convention
    assert cfg.optimizer == "kl-diag-polar-lora"
    assert cfg.beta1 == 0.95 and cfg.polar_method == "polar_express"
    assert cfg.muon_ns_steps == 8 and cfg.cw_picard_iters == 1
    assert cfg.precond_delta == 1e-4 and cfg.precond_refresh_every == 10
    assert cfg.precond_method == "eigh"           # QR today; Phase-1 adds higham
    assert "kl-diag-polar-lora" in cfg.banner()


def test_build_optimizer_kwargs_construct_and_forward():
    """The harness's production kwargs must construct the real optimizer AND the
    flags must actually take effect (ties the harness to the forwarding guardrail)."""
    cfg = protagonist_config("m", 16)
    opt = build_optimizer(_Model(), optimizer_type=cfg.optimizer, lr=1e-3,
                          **cfg.build_optimizer_kwargs())
    assert opt.beta1 == 0.95
    assert opt.ns_steps == 8
    assert opt.polar_method == "polar_express"
    assert opt.cw_nesterov is True
    assert opt.precond_refresh_every == 10


def test_build_optimizer_kwargs_precond_method_override():
    cfg = protagonist_config("m", 16)
    kw = cfg.build_optimizer_kwargs(precond_method="higham")
    assert kw["precond_method"] == "higham"


def test_record_bench_roundtrip(tmp_path):
    reg = tmp_path / "timing_registry.json"
    result = {
        "model_name": "allenai/OLMo-2-0425-1B", "lora_r": 256,
        "optimizer": "kl-diag-polar-lora", "muon_ns_steps": 8, "cw_picard_iters": 1,
        "batch_size": 4, "grad_accum_steps": 4, "max_seq_length": 2048,
        "compile_mode": "default", "bf16": True, "precond_method": "eigh",
        "mean_sec_per_step": 1.234, "n_reps": 40,
    }
    entry = record_bench(result, "blackwell", path=reg)
    assert entry["key"] == "blackwell/OLMo-2-0425-1B/r256/kl-diag-polar-lora/ns8/k1"
    assert entry["source"] == "bench_step_wall" and entry["wall_inclusive"] is False
    hit = timing.lookup(entry["key"], path=reg)
    assert hit is not None and hit[0] == 1.234


def test_record_bench_does_not_clobber_train_log(tmp_path):
    reg = tmp_path / "timing_registry.json"
    key = "blackwell/OLMo-2-0425-1B/r256/kl-diag-polar-lora/ns8/k1"
    # an authoritative (wall-inclusive) train-log entry already present
    timing.save_registry([{"key": key, "s_per_step": 6.0, "source": "train_log"}], reg)
    result = {"model_name": "allenai/OLMo-2-0425-1B", "lora_r": 256,
              "optimizer": "kl-diag-polar-lora", "muon_ns_steps": 8, "cw_picard_iters": 1,
              "mean_sec_per_step": 1.0}
    entry = record_bench(result, "blackwell", path=reg)
    assert entry["s_per_step"] == 6.0 and entry.get("source") == "train_log"
    # ...unless forced
    forced = record_bench(result, "blackwell", path=reg, force=True)
    assert forced["s_per_step"] == 1.0 and forced["source"] == "bench_step_wall"
