from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


LAYERS = 58
EXPERTS = 256
TOPK = 8


@dataclass(frozen=True)
class TraceRoute:
    request_index: int
    relative_path: str
    route: np.ndarray


class MoETraceRouteStore:
    """Small request-level route store for DeepSeek-R1-AWQ no-pruning replay."""

    def __init__(
        self,
        root: str | Path,
        workload: str,
        max_requests: int,
        profile_max_requests: int,
    ) -> None:
        self._root = Path(root)
        if not (self._root / "manifests" / "requests.csv").exists():
            raise FileNotFoundError(
                f"moe_trace_path must point at an R1 replay directory: {self._root}"
            )
        self._workload = workload
        self._max_requests = max_requests
        self._profile_max_requests = profile_max_requests
        self._routes: list[TraceRoute] = []
        self._profile = np.zeros((LAYERS, EXPERTS), np.int64)
        self._owners: dict[int, np.ndarray] = {}
        self._limited_owners: dict[tuple[int, int], np.ndarray] = {}
        self._load()

    @property
    def routes(self) -> list[TraceRoute]:
        return self._routes

    @property
    def profile(self) -> np.ndarray:
        return self._profile

    def owners(self, moe_gpus: int) -> np.ndarray:
        cached = self._owners.get(moe_gpus)
        if cached is None:
            cached = np.stack(
                [static_owners(self._profile[layer], moe_gpus) for layer in range(LAYERS)]
            )
            self._owners[moe_gpus] = cached
        return cached

    def limited_owners(self, hbm_gpus: int, experts_per_gpu: int) -> np.ndarray:
        key = (hbm_gpus, experts_per_gpu)
        cached = self._limited_owners.get(key)
        if cached is None:
            cached = np.stack(
                [
                    limited_static_owners(self._profile[layer], hbm_gpus, experts_per_gpu)
                    for layer in range(LAYERS)
                ]
            )
            self._limited_owners[key] = cached
        return cached

    def counts_for(
        self,
        request_ids: Iterable[int],
        decode_steps: Iterable[int],
        layer: int,
    ) -> np.ndarray:
        counts = np.zeros(EXPERTS, np.int32)
        if not self._routes:
            return counts
        routes = self._routes
        for request_id, decode_step in zip(request_ids, decode_steps):
            route = routes[int(request_id) % len(routes)].route
            step = min(max(int(decode_step), 0), route.shape[0] - 1)
            counts += np.bincount(route[step, layer], minlength=EXPERTS).astype(np.int32)
        return counts

    def selected_for(self, request_id: int, decode_step: int, layer: int) -> np.ndarray:
        route = self._routes[int(request_id) % len(self._routes)].route
        step = min(max(int(decode_step), 0), route.shape[0] - 1)
        return route[step, layer]

    def _selected_rows(self) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
        test_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
        train_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
        route_count = 0
        profile_count = 0
        with (self._root / "manifests" / "requests.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                matches_workload = self._workload == "mixed" or row["workload"] == self._workload
                if not matches_workload:
                    continue
                if row["split"] == "test" and route_count < self._max_requests:
                    test_by_shard[row["shard"]].append(row)
                    route_count += 1
                elif row["split"] == "train" and (
                    self._profile_max_requests <= 0
                    or profile_count < self._profile_max_requests
                ):
                    train_by_shard[row["shard"]].append(row)
                    profile_count += 1
        if not test_by_shard:
            raise ValueError(f"no held-out R1 routes found for workload {self._workload!r}")
        if not train_by_shard:
            raise ValueError(f"no train R1 routes found for workload {self._workload!r}")
        return test_by_shard, train_by_shard

    def _load(self) -> None:
        test_by_shard, train_by_shard = self._selected_rows()
        shard_names = sorted(set(test_by_shard) | set(train_by_shard))
        shard_root = self._root / "normalized" / "shards"
        for shard_name in shard_names:
            with np.load(shard_root / shard_name, allow_pickle=False) as shard:
                ids = shard["expert_ids"]
                offsets = shard["request_offsets"]
                prompt_lengths = shard["prompt_lengths"]
                decode_lengths = shard["decode_lengths"]

                for row in train_by_shard.get(shard_name, []):
                    idx = int(row["shard_request_index"])
                    start = int(offsets[idx]) + int(prompt_lengths[idx])
                    end = start + int(decode_lengths[idx])
                    data = ids[start:end]
                    for layer in range(LAYERS):
                        self._profile[layer] += np.bincount(
                            data[:, layer].reshape(-1), minlength=EXPERTS
                        )

                for row in test_by_shard.get(shard_name, []):
                    idx = int(row["shard_request_index"])
                    start = int(offsets[idx]) + int(prompt_lengths[idx])
                    end = start + int(decode_lengths[idx])
                    route = ids[start:end].copy()
                    if route.ndim != 3 or route.shape[1:] != (LAYERS, TOPK):
                        raise ValueError(
                            f"bad route shape for request {row['request_index']}: {route.shape}"
                        )
                    self._routes.append(
                        TraceRoute(
                            request_index=int(row["request_index"]),
                            relative_path=row["relative_path"],
                            route=route,
                        )
                    )


def static_owners(profile: np.ndarray, moe_gpus: int) -> np.ndarray:
    base_capacity, extra = divmod(EXPERTS, moe_gpus)
    remaining = np.full(moe_gpus, base_capacity, np.int64)
    if extra:
        remaining[:extra] += 1
    loads = np.zeros(moe_gpus, np.int64)
    owners = np.full(EXPERTS, -1, np.int16)
    for expert in np.lexsort((np.arange(EXPERTS), -profile)):
        candidates = np.flatnonzero(remaining > 0)
        gpu = int(candidates[np.argmin(loads[candidates])])
        owners[expert] = gpu
        remaining[gpu] -= 1
        loads[gpu] += int(profile[expert])
    return owners




def limited_static_owners(
    profile: np.ndarray, hbm_gpus: int, experts_per_gpu: int
) -> np.ndarray:
    owners = np.full(EXPERTS, -1, np.int16)
    if hbm_gpus <= 0 or experts_per_gpu <= 0:
        return owners

    remaining = np.full(hbm_gpus, experts_per_gpu, np.int64)
    loads = np.zeros(hbm_gpus, np.int64)
    for expert in np.lexsort((np.arange(EXPERTS), -profile)):
        candidates = np.flatnonzero(remaining > 0)
        if candidates.size == 0:
            break
        gpu = int(candidates[np.argmin(loads[candidates])])
        owners[expert] = gpu
        remaining[gpu] -= 1
        loads[gpu] += int(profile[expert])
    return owners


def assign_hybrid_experts(
    counts: np.ndarray,
    moe_gpus: int,
    hbf_gpus: int,
    hbm_static_owner: np.ndarray,
    previous: np.ndarray | None,
    cost_per_expert: float,
    cost_per_token: float,
) -> tuple[np.ndarray, float | None]:
    active = np.flatnonzero(counts)
    owners = np.full(EXPERTS, -1, np.int16)
    if active.size == 0:
        return owners, None

    hbf_gpus = min(max(int(hbf_gpus), 0), moe_gpus)
    hbm_gpus = moe_gpus - hbf_gpus
    hbf_devices = np.arange(hbm_gpus, moe_gpus, dtype=np.int16)
    score = np.where(counts > 0, cost_per_expert + cost_per_token * counts, 0.0)
    loads = np.zeros(moe_gpus, float)

    for expert in active[np.lexsort((active, -counts[active], -score[active]))]:
        static_gpu = int(hbm_static_owner[int(expert)])
        if hbf_gpus == moe_gpus:
            candidates = np.arange(moe_gpus, dtype=np.int16)
        elif hbf_gpus == 0:
            if static_gpu < 0:
                continue
            candidates = np.array([static_gpu], dtype=np.int16)
        elif static_gpu >= 0:
            candidates = np.concatenate((np.array([static_gpu], dtype=np.int16), hbf_devices))
        else:
            candidates = hbf_devices

        candidate_loads = loads[candidates]
        min_load = float(candidate_loads.min())
        candidates = candidates[np.isclose(candidate_loads, min_load)]
        old = -1 if previous is None else int(previous[expert])
        if old in candidates:
            gpu = old
        elif static_gpu in candidates:
            gpu = static_gpu
        else:
            gpu = int(candidates[0])
        owners[int(expert)] = gpu
        loads[gpu] += float(score[int(expert)])

    churn = None
    if previous is not None:
        common = active[(previous[active] >= 0) & (owners[active] >= 0)]
        if common.size:
            churn = float(np.mean(owners[common] != previous[common]))
    return owners, churn

def assign_experts(
    counts: np.ndarray,
    moe_gpus: int,
    policy: str,
    static_owner: np.ndarray | None,
    previous: np.ndarray | None,
    cost_per_expert: float,
    cost_per_token: float,
) -> tuple[np.ndarray, float | None]:
    active = np.flatnonzero(counts)
    if static_owner is not None:
        owners = np.full(EXPERTS, -1, np.int16)
        owners[active] = static_owner[active]
        return owners, None

    owners = np.full(EXPERTS, -1, np.int16)
    if policy == "active_count_dynamic":
        score = np.where(counts > 0, 1.0, 0.0)
    elif policy == "token_count_dynamic":
        score = counts.astype(float)
    elif policy == "cost_dynamic":
        score = np.where(counts > 0, cost_per_expert + cost_per_token * counts, 0.0)
    else:
        raise ValueError(f"unknown moe_trace_policy: {policy}")

    loads = np.zeros(moe_gpus, float)
    for expert in active[np.lexsort((active, -counts[active], -score[active]))]:
        min_load = loads.min()
        candidates = np.flatnonzero(np.isclose(loads, min_load))
        old = -1 if previous is None else int(previous[expert])
        gpu = old if old in candidates else int(candidates[0])
        owners[expert] = gpu
        loads[gpu] += float(score[expert])

    churn = None
    if previous is not None:
        common = active[previous[active] >= 0]
        if common.size:
            churn = float(np.mean(owners[common] != previous[common]))
    return owners, churn
