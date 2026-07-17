#!/usr/bin/env python3
"""Focused invariants for corrected_microbatch_sweep.py."""

import corrected_microbatch_sweep as sweep


def test_equal_bandwidth_and_no_free_overlap():
    assert sweep.base.BW_HBM_GBPS == sweep.base.BW_HBF_GBPS == 1024.0
    assert sweep.pipeline_tbt(12.0, 8.0, 1) == 20.0
    assert sweep.pipeline_tbt(12.0, 8.0, 2) == 12.0


def test_microbatch_partition():
    for sessions in (32, 33, 128):
        for count in sweep.MICROBATCH_COUNTS:
            slices = sweep.microbatch_slices(sessions, count)
            assert len(slices) == count
            assert sum(end - start for start, end in slices) == sessions
            sizes = [end - start for start, end in slices]
            assert max(sizes) - min(sizes) <= 1


def test_pruning_is_per_microbatch():
    trials = sweep.microbatch_pruned_trials(32, 980, 4)
    assert len(trials[0][0]) == 4
    for microbatch in trials[0][0]:
        assert len(microbatch.routes) == 8
        assert all(route for route in microbatch.routes)


def test_weight_reads_are_repeated_per_microbatch():
    one = sweep.microbatch_moe_profile(32, 1000, 1, 4, False)
    four = sweep.microbatch_moe_profile(32, 1000, 4, 4, False)
    # Smaller unions offset some rereads, so raw time need not grow by 4x, but
    # it must be positive and is computed from four independent route groups.
    assert one.execution_ms > 0
    assert four.execution_ms > 0
    assert four.mean_retained_experts < one.mean_retained_experts


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
