#!/usr/bin/env python3
"""Regression tests for HBF decode-attention KV bandwidth accounting."""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vidur.execution_time_predictor.hbf_execution_time_predictor import (
    HBFLinearRegressionExecutionTimePredictor,
)


def make_predictor():
    predictor = object.__new__(HBFLinearRegressionExecutionTimePredictor)
    predictor._num_layers_per_pipeline_stage = 61
    predictor._block_size = 16
    # DeepSeek-V3 MLA: 2(K,V) x 16 tokens x 9 KV heads x 32 dim x 2B / A=16.
    predictor._kv_block_bytes = 1152.0
    predictor._hbf_kv_read_bandwidth_bpns = 1024.0
    predictor._hbm_bandwidth_bpns = 1024.0
    predictor._config = SimpleNamespace(
        sparsity_fraction=1.0,
        hbm_kv_fraction=0.0,
        cached_context_tokens=0,
        naive_sparse=False,
    )
    return predictor


def decode_attention_ms(predictor, context_tokens: int, sessions: int) -> float:
    predictor._decode_reqs = lambda batch: [
        SimpleNamespace(num_processed_tokens=context_tokens) for _ in range(sessions)
    ]
    return predictor._get_attention_decode_execution_time(object())


def test_hbf_decode_attention_scales_with_context():
    predictor = make_predictor()
    short = decode_attention_ms(predictor, 8192, 64)
    long = decode_attention_ms(predictor, 131072, 64)
    assert long > short
    assert abs((long / short) - 16.0) < 1e-9


def test_hbf_decode_attention_scales_with_sessions():
    predictor = make_predictor()
    small = decode_attention_ms(predictor, 32768, 8)
    large = decode_attention_ms(predictor, 32768, 128)
    assert large > small
    assert abs((large / small) - 16.0) < 1e-9


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
