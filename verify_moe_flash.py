#!/usr/bin/env python3
"""
Verification for the MoE-experts-on-HBF path (vidur.memory_backends.moe_flash
+ HBFLinearRegressionExecutionTimePredictor MoE hooks).

Three layers of checks, cheapest first:

  1. REFERENCE EQUALITY — a verbatim copy of the formulas from
     attn_moe_batch_sweep.py (branch attn-ffn-batch-sweep) is embedded below
     as the reference implementation; moe_flash must match it bit-for-bit
     over a parameter grid.

  2. GOLDEN CSV — reproduce the committed sweep output
     attn-ffn-batch-sweep:sweep_out/tests/qwen3_moe_a3b/qwen3_moe_128k/
     attn_moe_sweep.csv (Qwen3-30B-A3B defaults: L=48 D=2048 E=128 k=8
     He=768 fp16, A100 312 TFLOPs, configs/hbf_paper.toml) to <1e-9 rel.

  3. PREDICTOR WIRING — call the predictor's _moe_layer_time_ms with the
     same geometry (via a stub self, no sklearn training needed) and check
     it equals the golden per-step values; plus invariants (monotonicity,
     routing saturation, fp8 halving, TP scaling, overlap=max).

Run:  PYTHONPATH=. <venv>/bin/python verify_moe_flash.py
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from vidur.memory_backends import moe_flash
from vidur.memory_backends.hbf.address_mapper import PlacementPolicy
from vidur.memory_backends.hbf.backend import HBFSimBackend

HBF_TOML = os.path.join(REPO, "configs", "hbf_paper.toml")

PASS = 0


def check(name, ok, detail=""):
    global PASS
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)
    PASS += 1


def rel_eq(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Reference implementation — copied VERBATIM from attn_moe_batch_sweep.py
#    (branch attn-ffn-batch-sweep). Do not "clean up"; it is the ground truth.
# ═══════════════════════════════════════════════════════════════════════════

def ref_unique_experts_uniform(B, E, k):
    # expected distinct experts under uniform-random routing (balls in bins)
    return E * (1.0 - (1.0 - k / E) ** B)


def ref_unique_experts_worst(B, E, k):
    return min(B * k, E)


def ref_flash_time_bw_ms(backend, total_bytes):
    if total_bytes <= 0:
        return 0.0
    bpns = backend.peak_bandwidth_gbps()  # bytes/ns
    return (total_bytes / bpns) / 1e6


def ref_moe_terms(backend, B, E, k, D, He, be, shared_He, fp16_tflops, overlap):
    """The sweep script's per-layer MoE math, assembled exactly as in its
    main() loop (uniform routing)."""
    per_expert_bytes = (2 * D * He + He * D) * be          # = 3*D*He*be
    expert_flops_per_token = 6 * D * He
    shared_bytes = 3 * D * shared_He * be
    shared_flops = 6 * D * shared_He
    flops_per_ms = fp16_tflops * 1e12 / 1e3

    nu = ref_unique_experts_uniform(B, E, k)
    flash = ref_flash_time_bw_ms(backend, nu * per_expert_bytes + shared_bytes)
    compute = (B * k * expert_flops_per_token + B * shared_flops) / flops_per_ms
    step = max(flash, compute) if overlap else flash + compute
    return nu, flash, compute, step


def main():
    backend = HBFSimBackend(
        config_path=HBF_TOML,
        placement_policy=PlacementPolicy.STRIPE_ACROSS_PLANES,
        num_layers=48,
    )

    # ───────────────────────────────────────────────────────────────────────
    # 1. Grid equality: moe_flash == reference, bit-for-bit
    # ───────────────────────────────────────────────────────────────────────
    grid = [
        (B, E, k, D, He, be, sh, ov)
        for B in (1, 3, 8, 64, 1000)
        for E, k in ((128, 8), (64, 2), (256, 1))
        for D, He in ((2048, 768), (7168, 2048))
        for be in (1, 2)
        for sh in (0, 512)
        for ov in (False, True)
    ]
    for B, E, k, D, He, be, sh, ov in grid:
        nu_r, fl_r, co_r, st_r = ref_moe_terms(
            backend, B, E, k, D, He, be, sh, 312.0, ov)
        nu = moe_flash.unique_experts_uniform(B, E, k)
        fl = moe_flash.flash_time_bw_ms(
            backend, moe_flash.moe_flash_bytes(B, E, k, D, He, be, sh))
        co = moe_flash.moe_compute_ms(
            B, k, D, He, moe_flash.flops_per_ms(312.0), sh)
        st = moe_flash.combine(fl, co, ov)
        assert nu == nu_r and fl == fl_r and co == co_r and st == st_r, (
            f"mismatch at {(B, E, k, D, He, be, sh, ov)}: "
            f"{(nu, fl, co, st)} != {(nu_r, fl_r, co_r, st_r)}")
        wu = moe_flash.unique_experts_worst(B, E, k)
        assert wu == ref_unique_experts_worst(B, E, k)
    check("grid equality: moe_flash == sweep-script reference",
          True, f"{len(grid)} points, exact")

    # ───────────────────────────────────────────────────────────────────────
    # 2. Golden CSV reproduction (Qwen3-30B-A3B defaults)
    #    from attn-ffn-batch-sweep:sweep_out/tests/qwen3_moe_a3b/
    #         qwen3_moe_128k/attn_moe_sweep.csv
    #    columns: B, nu, nworst, flash_u, flash_w, compute, step_u, step_w
    #    (all *_ms values are per decode step, full model, L=48)
    # ───────────────────────────────────────────────────────────────────────
    GOLDEN = [
        (1, 8.0, 8, 4.718592, 4.718592, 0.011614995692307693,
         4.730206995692308, 4.730206995692308),
        (2, 15.5, 16, 9.142272, 9.437184, 0.023229991384615387,
         9.165501991384616, 9.460413991384616),
        (4, 29.123046875, 32, 17.177472, 18.874368, 0.04645998276923077,
         17.223931982769233, 18.920827982769232),
        (8, 51.61990734934807, 64, 30.446660232421877, 37.748736,
         0.09291996553846155, 30.53958019796034, 37.841655965538465),
        (16, 82.42251130217052, 128, 48.61477530629142, 75.497472,
         0.1858399310769231, 48.80061523736835, 75.68331193107693),
        (32, 111.77103534374396, 128, 65.92523915058844, 75.497472,
         0.3716798621538462, 66.29691901274228, 75.86915186215384),
        (64, 125.94234926707806, 128, 74.28382021410505, 75.497472,
         0.7433597243076924, 75.02717993841273, 76.2408317243077),
        (128, 127.96692244891645, 128, 75.47796206650969, 75.497472,
         1.4867194486153847, 76.96468151512508, 76.98419144861539),
        (256, 127.99999145215324, 128, 75.49746695827483, 75.497472,
         2.9734388972307695, 78.4709058555056, 78.47091089723077),
    ]
    L, D, He, E, k, be = 48, 2048, 768, 128, 8, 2
    fpm = moe_flash.flops_per_ms(312.0)
    for row in GOLDEN:
        B, g_nu, g_nw, g_flu, g_flw, g_co, g_stu, g_stw = row
        nu = moe_flash.unique_experts_uniform(B, E, k)
        nw = moe_flash.unique_experts_worst(B, E, k)
        flu = moe_flash.flash_time_bw_ms(
            backend, moe_flash.moe_flash_bytes(B, E, k, D, He, be)) * L
        flw = moe_flash.flash_time_bw_ms(
            backend, moe_flash.moe_flash_bytes(B, E, k, D, He, be, worst=True)) * L
        co = moe_flash.moe_compute_ms(B, k, D, He, fpm) * L
        stu = moe_flash.combine(flu / L, co / L, False) * L
        stw = moe_flash.combine(flw / L, co / L, False) * L
        for name, got, want in (
            ("nu", nu, g_nu), ("nworst", nw, g_nw),
            ("flash_u", flu, g_flu), ("flash_w", flw, g_flw),
            ("compute", co, g_co), ("step_u", stu, g_stu),
            ("step_w", stw, g_stw),
        ):
            assert rel_eq(got, want), \
                f"B={B} {name}: got {got!r}, golden {want!r}"
    check("golden CSV: committed Qwen3-30B-A3B sweep reproduced",
          True, f"{len(GOLDEN)} rows x 7 cols, rel<1e-9")

    # ───────────────────────────────────────────────────────────────────────
    # 3. Predictor wiring: _moe_layer_time_ms via a stub self
    # ───────────────────────────────────────────────────────────────────────
    from vidur.execution_time_predictor.hbf_execution_time_predictor import (
        HBFLinearRegressionExecutionTimePredictor as P,
    )

    class StubModelConfig:
        embedding_dim = D

    class StubBatch:
        def __init__(self, n_decode):
            # decode: one token per sequence per step
            self.num_tokens = [1] * n_decode

    class Stub:
        _model_config = StubModelConfig()
        _hbf = backend
        _moe_num_experts = E
        _moe_top_k = k
        _moe_intermediate = He
        _moe_dtype_bytes = be
        _moe_shared_intermediate = 0
        _moe_overlap = False
        _moe_routing_worst = False
        _moe_use_sched = False
        _moe_flops_per_ms = fpm
        _moe_tp_size = 1

    stub = Stub()
    for row in GOLDEN:
        B, _, _, _, _, _, g_stu, _ = row
        got = P._moe_layer_time_ms(stub, StubBatch(B)) * L
        assert rel_eq(got, g_stu), f"B={B}: predictor {got!r} != golden {g_stu!r}"
    check("predictor _moe_layer_time_ms == golden step (uniform, serial)", True)

    # worst-case routing
    stub._moe_routing_worst = True
    for row in GOLDEN:
        B, _, _, _, _, _, _, g_stw = row
        got = P._moe_layer_time_ms(stub, StubBatch(B)) * L
        assert rel_eq(got, g_stw), f"B={B}: predictor {got!r} != golden {g_stw!r}"
    stub._moe_routing_worst = False
    check("predictor _moe_layer_time_ms == golden step (worst-case routing)", True)

    # ── invariants ─────────────────────────────────────────────────────────
    steps = [P._moe_layer_time_ms(stub, StubBatch(B)) for B, *_ in GOLDEN]
    check("monotonic: MoE step non-decreasing in batch size",
          all(steps[i] <= steps[i + 1] + 1e-12 for i in range(len(steps) - 1)))

    check("routing: E_unique(1) == top_k",
          abs(moe_flash.unique_experts_uniform(1, E, k) - k) < 1e-9)
    check("routing: E_unique saturates to E",
          moe_flash.unique_experts_uniform(10**9, E, k) > E - 1e-3)

    # fp8 experts halve flash bytes exactly
    b16 = moe_flash.moe_flash_bytes(8, E, k, D, He, 2)
    b8 = moe_flash.moe_flash_bytes(8, E, k, D, He, 1)
    check("fp8 experts = exactly half the fp16 flash bytes", b16 == 2 * b8)

    # TP sharding: tp=4 quarters the per-GPU MoE layer time
    t1 = P._moe_layer_time_ms(stub, StubBatch(8))
    stub._moe_tp_size = 4
    t4 = P._moe_layer_time_ms(stub, StubBatch(8))
    stub._moe_tp_size = 1
    check("TP sharding: tp=4 gives exactly 1/4 the per-GPU MoE time",
          rel_eq(t1, 4 * t4))

    # overlap: max instead of sum
    stub._moe_overlap = True
    t_ov = P._moe_layer_time_ms(stub, StubBatch(8))
    stub._moe_overlap = False
    fl = moe_flash.flash_time_bw_ms(
        backend, moe_flash.moe_flash_bytes(8, E, k, D, He, be))
    co = moe_flash.moe_compute_ms(8, k, D, He, fpm)
    check("overlap=True gives max(flash, compute)",
          rel_eq(t_ov, max(fl, co)) and rel_eq(t1, fl + co))

    # empty batch → 0
    check("empty batch → 0 ms", P._moe_layer_time_ms(stub, StubBatch(0)) == 0.0)

    # sched path runs and is faster than (or equal to) the bw bound
    stub._moe_use_sched = True
    t_sched = P._moe_layer_time_ms(stub, StubBatch(8))
    stub._moe_use_sched = False
    check("plane-scheduler flash path runs; sched <= bw bound (tRC fast path)",
          0.0 < t_sched <= t1, f"sched={t_sched:.4f}ms bw={t1:.4f}ms")

    print(f"\nAll {PASS} checks passed.")


if __name__ == "__main__":
    main()
