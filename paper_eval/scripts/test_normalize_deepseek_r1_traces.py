#!/usr/bin/env python3
"""Regression tests for DeepSeek-R1 trace normalization."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from normalize_deepseek_r1_traces import (
    TraceValidationError,
    convert,
    deterministic_split,
    normalize_request,
)


def complete_entry(tokens: int) -> dict[str, object]:
    entry: dict[str, object] = {str(layer): None for layer in range(3)}
    for layer in range(3, 61):
        entry[str(layer)] = [
            [int((layer + token + offset) % 256) for offset in range(8)]
            for token in range(tokens)
        ]
    return entry


class NormalizeRequestTests(unittest.TestCase):
    def write_trace(self, directory: Path, entries: list[dict[str, object]]) -> Path:
        path = directory / "trace.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_discards_only_prefix_truncated_final_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            truncated = complete_entry(1)
            for layer in range(56, 61):
                del truncated[str(layer)]
            normalized = normalize_request(
                self.write_trace(directory, [complete_entry(2), complete_entry(1), truncated])
            )

        self.assertEqual(normalized.expert_ids.shape, (3, 58, 8))
        self.assertEqual(normalized.expert_ids.dtype, np.uint8)
        self.assertEqual(normalized.prompt_tokens, 2)
        self.assertEqual(normalized.raw_decode_entries, 2)
        self.assertEqual(normalized.valid_decode_tokens, 1)
        self.assertEqual(normalized.discard["highest_present_layer"], 55)

    def test_discards_null_suffix_in_final_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            truncated = complete_entry(1)
            truncated["60"] = None
            normalized = normalize_request(
                self.write_trace(directory, [complete_entry(2), truncated])
            )

        self.assertEqual(normalized.expert_ids.shape, (2, 58, 8))
        self.assertEqual(normalized.valid_decode_tokens, 0)
        self.assertEqual(normalized.discard["highest_present_layer"], 59)

    def test_rejects_prefix_truncation_before_final_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            truncated = complete_entry(1)
            del truncated["60"]
            path = self.write_trace(
                directory,
                [complete_entry(2), truncated, complete_entry(1)],
            )
            with self.assertRaises(TraceValidationError):
                normalize_request(path)

    def test_rejects_duplicate_expert_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            invalid = complete_entry(1)
            invalid["3"][0][1] = invalid["3"][0][0]
            with self.assertRaises(TraceValidationError):
                normalize_request(self.write_trace(directory, [invalid]))

    def test_split_is_deterministic(self) -> None:
        first = deterministic_split("mmlu/topic/0.json", "seed", 0.2)
        second = deterministic_split("mmlu/topic/0.json", "seed", 0.2)
        self.assertEqual(first, second)


class ConversionTests(unittest.TestCase):
    def test_writes_offsets_manifests_and_uint8_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            category = input_root / "mmlu" / "topic"
            category.mkdir(parents=True)
            (category / "0.json").write_text(
                json.dumps([complete_entry(2), complete_entry(1)]), encoding="utf-8"
            )
            (category / "1.json").write_text(
                json.dumps([complete_entry(3), complete_entry(1)]), encoding="utf-8"
            )
            output_dir = root / "output"
            args = argparse.Namespace(
                input_root=input_root,
                output_dir=output_dir,
                requests_per_shard=2,
                test_fraction=0.2,
                split_seed="test-seed",
                limit=None,
                replace_generated=False,
                progress_every=0,
            )
            summary = convert(args)

            with np.load(
                output_dir / "normalized" / "shards" / "shard_00000.npz",
                allow_pickle=False,
            ) as shard:
                self.assertEqual(shard["expert_ids"].dtype, np.uint8)
                self.assertEqual(shard["expert_ids"].shape, (7, 58, 8))
                np.testing.assert_array_equal(shard["request_offsets"], [0, 3, 7])
                np.testing.assert_array_equal(shard["prompt_lengths"], [2, 3])
                np.testing.assert_array_equal(shard["decode_lengths"], [1, 1])

            with (output_dir / "manifests" / "requests.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["totals"]["requests"], 2)
            self.assertEqual(summary["totals"]["routed_token_records"], 7)


if __name__ == "__main__":
    unittest.main()
