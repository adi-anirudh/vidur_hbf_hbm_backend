#!/usr/bin/env python3
"""Focused invariants for corrected_ctx_workload_sweep.py."""

import importlib.util
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("corrected_ctx_workload_sweep.py")
SPEC = importlib.util.spec_from_file_location("corrected_sweep", MODULE_PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)


def test_geometry_and_capacity():
    assert SWEEP.KV_BYTES_PER_TOKEN_LAYER == 576
    assert SWEEP.N_MOE == 58
    assert math.isclose(SWEEP.gb(SWEEP.ROUTED_EXPERT_BYTES_TOTAL), 653.908770816)
    assert SWEEP.moe_capacity_ok(1, dynamic=True)
    assert SWEEP.attention_capacity_sessions(8192, SWEEP.HBM_CAP_GB) > 100
    assert SWEEP.attention_capacity_sessions(131072, SWEEP.HBM_CAP_GB) < 20


def test_fixed_placement_is_balanced():
    for m in (3, 8, 16, 32):
        owner = SWEEP.fixed_owner(7, m)
        shard_sizes = [owner.count(gpu) for gpu in range(m)]
        assert max(shard_sizes) - min(shard_sizes) <= 1
        assert sum(shard_sizes) == SWEEP.N_EXPERTS


def test_pruning_top1_and_monotonicity():
    layer = SWEEP.routing_samples(32)[0][0]
    full = SWEEP.prune_layer(layer, 1.0)
    p98 = SWEEP.prune_layer(layer, 0.98)
    p90 = SWEEP.prune_layer(layer, 0.90)
    assert len(p90.retained) <= len(p98.retained) <= len(full.retained)
    retained90 = set(p90.retained)
    for experts, weights in zip(layer.routes, layer.gate_weights):
        assert experts[max(range(len(weights)), key=lambda i: weights[i])] in retained90
    assert all(routes for routes in p90.routes)


def test_dynamic_assignment_is_integral_and_complete():
    layer = SWEEP.pruned_samples(64, 980)[0][0]
    owner = SWEEP.dynamic_lpt_owner(layer, 8, SWEEP.BW_HBF_GBPS)
    assert all(0 <= owner[e] < 8 for e in layer.retained)
    assert all(owner[e] == -1 for e in set(range(SWEEP.N_EXPERTS)) - set(layer.retained))


def test_attention_time_and_capacity_scale_with_context():
    short = SWEEP.attention_stage_ms(8192, 8, SWEEP.BW_HBM_GBPS)
    long = SWEEP.attention_stage_ms(131072, 8, SWEEP.BW_HBM_GBPS)
    assert long > short
    assert SWEEP.attention_capacity_sessions(8192, SWEEP.HBM_CAP_GB) > SWEEP.attention_capacity_sessions(131072, SWEEP.HBM_CAP_GB)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
