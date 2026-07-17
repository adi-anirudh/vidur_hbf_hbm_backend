#!/usr/bin/env python3
"""Regression tests for R1 fixed-cohort scheduling primitives."""

import unittest

import numpy as np

from replay_deepseek_r1_no_prune import (
    EXPERTS,
    static_owners,
    dynamic_event,
    static_event,
)


class SchedulingTests(unittest.TestCase):
    def test_static_placement_has_equal_capacity(self) -> None:
        profile = np.arange(EXPERTS, dtype=np.int64)
        owners = static_owners(profile, 16)
        np.testing.assert_array_equal(np.bincount(owners), np.full(16, 16))

    def test_dynamic_lpt_reaches_simple_lower_bound(self) -> None:
        counts = np.zeros(EXPERTS, dtype=np.int16)
        counts[:8] = [8, 7, 6, 5, 4, 3, 2, 1]
        maximum, fetched, owners, churn = dynamic_event(counts, 4)
        self.assertEqual(maximum, 9)
        self.assertEqual(fetched, 2)
        self.assertIsNone(churn)
        self.assertTrue(np.all(owners[:8] >= 0))

    def test_stability_preference_avoids_tie_churn(self) -> None:
        counts = np.zeros(EXPERTS, dtype=np.int16)
        counts[:8] = 1
        _, _, first, _ = dynamic_event(counts, 4)
        _, _, second, churn = dynamic_event(counts, 4, first)
        self.assertEqual(churn, 0.0)
        np.testing.assert_array_equal(first, second)

    def test_static_event_counts_tokens_and_fetches(self) -> None:
        owners = np.repeat(np.arange(4, dtype=np.int16), 64)
        counts = np.zeros(EXPERTS, dtype=np.int16)
        counts[[0, 1, 64, 128, 192]] = [3, 2, 4, 5, 6]
        maximum, fetched = static_event(counts, owners, 4)
        self.assertEqual(maximum, 6)
        self.assertEqual(fetched, 2)


if __name__ == "__main__":
    unittest.main()
