"""
HBF-backed linear-regression execution time predictor.

Memory-bound operations are replaced by explicit memory models:

  HBF bandwidth model  →  cold KV cache reads (decode)
  HBM bank scheduler   →  KV cache writes (one new token per decode sequence)
  HBM bandwidth model  →  weight loads (QKV, O, MLP up/gate/down)

Operations that are compute-bound (prefill attention, norms, activations,
comms) remain in the parent sklearn model.

Weight loads use an analytical bandwidth model rather than the per-row bank
scheduler because sequential streaming access hits each HBM bank in burst
mode — per-row activate/precharge latency does not apply.  KV writes use the
bank scheduler because multiple sequences write to random addresses and can
conflict on the same bank.

Both weight loads and KV writes share the same RamulatorBackend, so that
cross-traffic bank conflicts are captured when both are submitted within a
single decode step.

Usage
-----
    predictor_config = HBFLinearRegressionExecutionTimePredictorConfig(
        hbfsim_config_path = "configs/staged_hbm_base_baseline.toml",
        placement_policy   = "STRIPE_ACROSS_PLANES",
        hot_kv_fraction    = 0.0,
        use_ramulator      = False,
    )
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vidur.config import (
    BaseReplicaSchedulerConfig,
    HBFLinearRegressionExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch
from vidur.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from vidur.execution_time_predictor.moe_trace import (
    EXPERTS as TRACE_EXPERTS,
    LAYERS as TRACE_LAYERS,
    MoETraceRouteStore,
    assign_experts,
    assign_hybrid_experts,
)
from vidur.memory_backends import (
    CompositeBackend,
    HBFSimBackend,
    PlacementPolicy,
    RamulatorBackend,
)
from vidur.memory_backends import moe_flash
from vidur.memory_backends.base import MemoryBackend, OpType
from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader
from vidur.logger import init_logger

logger = init_logger(__name__)


_POLICY_MAP = {
    "STRIPE_ACROSS_PLANES": PlacementPolicy.STRIPE_ACROSS_PLANES,
    "PACK_BY_SEQUENCE":     PlacementPolicy.PACK_BY_SEQUENCE,
    "INTERLEAVE_BY_TOKEN":  PlacementPolicy.INTERLEAVE_BY_TOKEN,
}

# Start of the weight address region in HBM.
# KV blocks use block_id × 64 KB; at 2M blocks that's ~128 GB.  512 GB is safe.
_WEIGHT_ADDR_BASE = 512 * (1 << 30)


class HBFLinearRegressionExecutionTimePredictor(LinearRegressionExecutionTimePredictor):
    """
    LinearRegression predictor for compute-bound ops; HBF/HBM backends for all
    memory-bound ops.

    Replaced by simulation (vs sklearn lookup table):
      _get_attention_layer_pre_proj_execution_time   QKV weight load  → HBM bandwidth
      _get_attention_layer_post_proj_execution_time  O   weight load  → HBM bandwidth
      _get_mlp_layer_up_proj_execution_time          up+gate weights  → HBM bandwidth
      _get_mlp_layer_down_proj_execution_time        down weights     → HBM bandwidth
      _get_attention_kv_cache_save_execution_time    new-token writes → HBM bank sched
      _get_attention_decode_execution_time           KV reads         → bandwidth model

    Kept in sklearn (compute-bound or negligible memory footprint):
      _get_attention_prefill_execution_time
      _get_attention_rope_execution_time
      _get_attn_norm / _get_mlp_norm / _get_mlp_layer_act / _get_add_layer_act
      _get_tensor_parallel_communication_time
      _get_pipeline_parallel_communication_time
    """

    def __init__(
        self,
        predictor_config: HBFLinearRegressionExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

        # ── HBF: cold KV reads via NAND plane scheduler ──────────────────
        policy = _POLICY_MAP.get(
            predictor_config.placement_policy,
            PlacementPolicy.STRIPE_ACROSS_PLANES,
        )
        self._hbf = HBFSimBackend(
            config_path=predictor_config.hbfsim_config_path,
            placement_policy=policy,
            num_layers=self._model_config.num_layers,
        )
        self._hbf_backend = self._hbf  # for plane_stats() access

        # ── HBM: weight loads + KV writes + hot KV reads ─────────────────
        self._hbm = RamulatorBackend(config_path=predictor_config.hbfsim_config_path)

        # _memory_backend drives submit_decode_step for the hot-KV path;
        # in cold-only mode it delegates to _hbf.
        if predictor_config.use_ramulator:
            self._memory_backend: MemoryBackend = CompositeBackend(
                hbf=self._hbf, hbm=self._hbm
            )
        else:
            self._memory_backend = self._hbf

        # ── HBM timing / bandwidth ────────────────────────────────────────
        hbf_cfg = HBFSimConfigLoader(predictor_config.hbfsim_config_path).load()
        # bandwidth_gbps is GBps (gigabytes/s); 1 GBps = 1 byte/ns
        self._hbm_bandwidth_bpns = hbf_cfg.hbm.bandwidth_gbps
        self._hbf_kv_read_bandwidth_bpns = (
            predictor_config.hbf_kv_read_bw_gbps
            if predictor_config.hbf_kv_read_bw_gbps > 0.0
            else self._hbm_bandwidth_bpns
        )

        # ── Model geometry ────────────────────────────────────────────────
        E   = self._model_config.embedding_dim
        H   = self._model_config.mlp_hidden_dim
        Nq  = self._model_config.num_q_heads
        Nkv = self._model_config.num_kv_heads
        D   = E // Nq  # head_dim

        # ── TP sharding scope ─────────────────────────────────────────────
        # 'experts': attention/KV modeled at full single-GPU cost (original
        # HBF-study behavior); only the MoE path divides by TP.
        # 'full': attention heads and KV are Megatron-sharded — each worker
        # holds Nq/TP query heads and ceil(Nkv/TP) KV heads (GQA heads
        # replicate when TP > Nkv, so per-worker KV does not shrink past
        # one head's worth).
        _tp = replica_config.tensor_parallel_size
        self._disagg_replicated = (
            predictor_config.disagg_num_moe_gpus > 0
            and predictor_config.disagg_expert_placement == "replicated"
        )
        if self._disagg_replicated:
            # Sequence-DP attention pool: each of the A GPUs keeps a FULL
            # weight copy and owns 1/A of the sequences (KV divides exactly
            # by A below); no head sharding, no all-reduce.
            self._attn_tp = 1
        else:
            self._attn_tp = _tp if predictor_config.tp_shard_scope == "full" else 1
        # ceil: uneven head counts put the extra head on some workers; the
        # step time is set by the largest shard.
        Nq_s  = math.ceil(Nq / self._attn_tp)
        Nkv_s = math.ceil(Nkv / self._attn_tp)

        # ── Weight sizes per layer per worker (FP16 = 2 bytes) ────────────
        # QKV: Q (E×Nq_sD) + K (E×Nkv_sD) + V (E×Nkv_sD)
        self._qkv_weight_bytes = (E * Nq_s * D + 2 * E * Nkv_s * D) * 2
        # O projection: Nq_sD × E (row-parallel)
        self._o_weight_bytes = Nq_s * D * E * 2
        # MLP gate fused with up in gated architectures (LLaMA SwiGLU)
        _gate = 2 if self._model_config.use_gated_mlp else 1
        self._mlp_up_weight_bytes   = E * H * _gate * 2 // self._attn_tp
        self._mlp_down_weight_bytes = H * E * 2 // self._attn_tp

        # ── KV sizes per worker ───────────────────────────────────────────
        # Full KV block: K+V, block_size tokens, FP16
        self._kv_block_bytes = 2 * self._block_size * Nkv_s * D * 2
        # One new token write per decode step: K+V, 1 token, FP16
        self._kv_write_bytes = 2 * Nkv_s * D * 2
        if self._disagg_replicated:
            # Sequence-DP: each attention GPU reads/writes only its own
            # sequences' KV — exact 1/A split (not head-granular).
            self._kv_block_bytes /= _tp
            self._kv_write_bytes /= _tp

        # ── Weight address layout (layer-major) in HBM ───────────────────
        # [layer0: QKV | O | up | down] [layer1: ...] ...
        self._off_qkv  = 0
        self._off_o    = self._qkv_weight_bytes
        self._off_up   = self._qkv_weight_bytes + self._o_weight_bytes
        self._off_down = (self._qkv_weight_bytes + self._o_weight_bytes
                          + self._mlp_up_weight_bytes)
        self._layer_weight_stride = (
            self._qkv_weight_bytes + self._o_weight_bytes
            + self._mlp_up_weight_bytes + self._mlp_down_weight_bytes
        )

        # ── Sparse-MoE geometry (expert weights on HBF flash) ─────────────
        # Active when num_experts > 1; math shared with attn_moe_batch_sweep.py
        # via vidur.memory_backends.moe_flash. Expert weights are tensor-parallel
        # sharded: each of the TP workers streams 1/TP of the activated expert
        # bytes from its own HBF stack and does 1/TP of the expert GEMM.
        self._moe_enabled = predictor_config.num_experts > 1
        self._moe_trace_store = None
        self._moe_trace_previous: dict[tuple[str, int], np.ndarray] = {}
        self._moe_trace_diagnostics_path = None
        if self._moe_enabled:
            self._moe_num_experts = predictor_config.num_experts
            self._moe_top_k = predictor_config.moe_top_k
            self._moe_intermediate = (
                predictor_config.moe_intermediate
                if predictor_config.moe_intermediate > 0
                else self._model_config.mlp_hidden_dim
            )
            self._moe_dtype_bytes = predictor_config.moe_dtype_bytes
            self._moe_shared_intermediate = (
                predictor_config.shared_expert_intermediate
            )
            self._moe_overlap = predictor_config.moe_overlap
            self._moe_routing_worst = predictor_config.moe_routing == "worst"
            self._moe_use_sched = predictor_config.moe_use_sched
            self._moe_weights_in_hbm = predictor_config.moe_weights_in_hbm
            self._moe_a2a_bw_gbps = predictor_config.moe_a2a_bw_gbps
            self._moe_flops_per_ms = moe_flash.flops_per_ms(
                replica_config.device_config.fp16_tflops
            )
            # Disaggregation: experts live on a dedicated pool of M GPUs;
            # tensor_parallel_size counts only the attention pool (A).
            self._disagg_moe_gpus = predictor_config.disagg_num_moe_gpus
            self._disagg_link_bw = predictor_config.disagg_link_bw_gbps
            self._disagg_overlap = predictor_config.disagg_microbatch_overlap
            self._disagg_affinity = predictor_config.disagg_routing == "affinity"
            self._disagg_clustered = predictor_config.disagg_routing == "clustered"
            self._disagg_cluster_eff = predictor_config.disagg_cluster_efficiency
            if self._disagg_moe_gpus > 0:
                self._moe_tp_size = self._disagg_moe_gpus
            else:
                self._moe_tp_size = replica_config.tensor_parallel_size

            if predictor_config.moe_trace_path:
                self._moe_trace_policy = predictor_config.moe_trace_policy
                self._moe_trace_hybrid_hbf_gpus = predictor_config.moe_trace_hybrid_hbf_gpus
                self._moe_trace_hybrid_hbm_experts_per_gpu = (
                    predictor_config.moe_trace_hybrid_hbm_experts_per_gpu
                )
                self._moe_trace_read_bw_bpns = predictor_config.moe_trace_read_bw_gbps
                self._moe_trace_store = MoETraceRouteStore(
                    root=predictor_config.moe_trace_path,
                    workload=predictor_config.moe_trace_workload,
                    max_requests=predictor_config.moe_trace_max_requests,
                    profile_max_requests=predictor_config.moe_trace_profile_max_requests,
                )
                if predictor_config.moe_trace_diagnostics_path:
                    self._moe_trace_diagnostics_path = Path(
                        predictor_config.moe_trace_diagnostics_path
                    )
                    self._moe_trace_diagnostics_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    self._moe_trace_diagnostics_path.write_text(
                        "", encoding="utf-8"
                    )
                logger.info(
                    "Loaded %d held-out MoE trace routes from %s for workload=%s "
                    "policy=%s",
                    len(self._moe_trace_store.routes),
                    predictor_config.moe_trace_path,
                    predictor_config.moe_trace_workload,
                    predictor_config.moe_trace_policy,
                )

        if predictor_config.enforce_memory_check:
            self._validate_memory_fit(replica_config, predictor_config)

        self._req_counter = 0

        # Exact memoization of the discrete-event backends. Both start from
        # a clean state every call (reset_step / drain), so results are pure
        # functions of their inputs: flash reads of pages-per-plane, KV
        # writes of the batch's request-id multiset (addresses derive from
        # request ids). Decode batch composition persists across consecutive
        # steps, so hit rates are high.
        self._flash_read_cache: dict = {}
        self._kv_write_cache: dict = {}

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        # Replicated-disagg: attention pool is sequence-DP (no head sharding)
        # and every MoE GPU is self-contained — no all-reduce anywhere.
        # Same for the colocated DP-attention + EP-experts baseline.
        if getattr(self, "_disagg_replicated", False):
            return 0.0
        if self._config.assume_dp_attention:
            return 0.0
        return super()._get_tensor_parallel_communication_time(batch)

    # ──────────────────────────────────────────────────────────────────────
    # Memory-limit validation
    # ──────────────────────────────────────────────────────────────────────

    def _validate_memory_fit(
        self,
        replica_config: ReplicaConfig,
        predictor_config: HBFLinearRegressionExecutionTimePredictorConfig,
    ) -> None:
        """
        Static per-GPU residency check against device capacities.

        HBM (per worker): attention weights for all pipeline-stage layers,
        plus dense MLP weights when MoE is disabled.  KV is excluded — its
        HBM share (hbm_kv_fraction) is a runtime working set bounded by the
        scheduler; the remaining headroom for it is reported.

        HBF flash (per worker): all E experts' weights (plus shared expert)
        for all layers, sharded 1/TP — experts must *reside* on flash in
        full, independent of how many are activated per step.  Cold KV also
        lives here; remaining headroom is reported.
        """
        GiB = 1 << 30
        num_layers = (
            TRACE_LAYERS
            if getattr(predictor_config, "moe_trace_path", "")
            else self._num_layers_per_pipeline_stage
        )

        hbm_capacity = (
            replica_config.device_config.total_memory_gb
            * GiB
            * (1 - replica_config.memory_margin_fraction)
        )
        hbm_weights = (self._qkv_weight_bytes + self._o_weight_bytes) * num_layers
        if not self._moe_enabled:
            hbm_weights += (
                self._mlp_up_weight_bytes + self._mlp_down_weight_bytes
            ) * num_layers

        hbf_capacity = self._hbf.total_capacity_bytes()
        hbf_experts = 0
        if self._moe_enabled:
            D = self._model_config.embedding_dim
            per_expert = moe_flash.per_expert_bytes(
                D, self._moe_intermediate, self._moe_dtype_bytes
            )
            shared = moe_flash.per_expert_bytes(
                D, self._moe_shared_intermediate, self._moe_dtype_bytes
            )
            expert_bytes = (self._moe_num_experts * per_expert + shared) * num_layers
            if not self._disagg_replicated:
                expert_bytes //= self._moe_tp_size  # replicated: full set per MoE GPU
            if self._moe_weights_in_hbm:
                hbm_weights += expert_bytes
            else:
                hbf_experts = expert_bytes

        disagg_m = getattr(self, "_disagg_moe_gpus", 0)
        if disagg_m > 0:
            # Disaggregated pools: KV lives on attention GPUs' stacks (full
            # flash headroom); experts live on the M MoE GPUs' stacks.
            logger.info(
                f"[HBF memory check] DISAGG attn_gpus="
                f"{self._replica_config.tensor_parallel_size} moe_gpus={disagg_m} "
                f"tp_shard_scope={predictor_config.tp_shard_scope} | "
                f"attn-GPU HBM: weights {hbm_weights / GiB:.1f} GiB of "
                f"{hbm_capacity / GiB:.1f} GiB usable | "
                f"attn-GPU HBF: {hbf_capacity / GiB:.1f} GiB for cold KV | "
                f"moe-GPU HBF: experts {hbf_experts / GiB:.1f} GiB of "
                f"{hbf_capacity / GiB:.1f} GiB"
            )
        else:
            logger.info(
                f"[HBF memory check] tp_shard_scope={predictor_config.tp_shard_scope} "
                f"attn_tp={self._attn_tp} moe_tp={getattr(self, '_moe_tp_size', 1)} | "
                f"HBM/GPU: weights {hbm_weights / GiB:.1f} GiB of "
                f"{hbm_capacity / GiB:.1f} GiB usable "
                f"({(hbm_capacity - hbm_weights) / GiB:.1f} GiB left for hot KV) | "
                f"HBF/GPU: experts {hbf_experts / GiB:.1f} GiB of "
                f"{hbf_capacity / GiB:.1f} GiB "
                f"({(hbf_capacity - hbf_experts) / GiB:.1f} GiB left for cold KV)"
            )
        if hbm_weights > hbm_capacity:
            raise ValueError(
                f"HBM-resident weights ({hbm_weights / GiB:.1f} GiB/GPU) exceed "
                f"usable HBM ({hbm_capacity / GiB:.1f} GiB). Increase "
                f"tensor_parallel_size with tp_shard_scope='full', add pipeline "
                f"stages, or move weights to HBF."
            )
        if hbf_experts > hbf_capacity:
            raise ValueError(
                f"HBF-resident expert weights ({hbf_experts / GiB:.1f} GiB/GPU) "
                f"exceed flash capacity ({hbf_capacity / GiB:.1f} GiB). Increase "
                f"tensor_parallel_size or flash capacity in the HBFSim TOML."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _has_decode(self, batch: Batch) -> bool:
        return any(r.is_prefill_complete for r in batch.requests)

    def _decode_reqs(self, batch: Batch):
        return [r for r in batch.requests if r.is_prefill_complete]

    # ──────────────────────────────────────────────────────────────────────
    # HBM weight load: analytical streaming bandwidth
    # ──────────────────────────────────────────────────────────────────────

    def _hbm_weight_load_ms(self, weight_bytes: int) -> float:
        """
        Analytical latency for a sequential weight-matrix streaming read.

        Sequential streaming accesses each HBM bank once in burst mode;
        per-row activate latency does not dominate.  The bandwidth formula
        is exact for this access pattern.

        Returns per-layer ms.
        """
        return (weight_bytes / self._hbm_bandwidth_bpns) / 1e6

    # ──────────────────────────────────────────────────────────────────────
    # HBM KV write: bank scheduler (random access per sequence)
    # ──────────────────────────────────────────────────────────────────────

    def _hbm_kv_write_ms(self, decode_reqs, num_layers: int) -> float:
        """
        Submit one KV write per (sequence × layer) to the HBM bank scheduler.

        Address: (seq_id × num_layers + layer_id) × 64 KB — identical to the
        KV read address scheme so writes and reads map to the same physical
        bank.  Multiple sequences writing to the same bank produce row conflicts
        that the scheduler models with tRP + tRCD + tCL.

        Returns per-pipeline-layer ms.
        """
        if not decode_reqs:
            return 0.0

        cache_key = tuple(sorted(r.id for r in decode_reqs))
        cached = self._kv_write_cache.get(cache_key)
        if cached is not None:
            return cached

        page = 64 * 1024  # 64 KB — matches RamulatorBackend KV address scheme
        for r in decode_reqs:
            for layer in range(num_layers):
                addr = (r.id * num_layers + layer) * page
                self._hbm.submit_at_addr(
                    req_id=self._next_req_id(),
                    op=OpType.WRITE,
                    addr=addr,
                    size_bytes=self._kv_write_bytes,
                )

        completions = self._hbm.drain()
        result = (
            (max(c.latency_ns for c in completions) / 1e6) / num_layers
            if completions
            else 0.0
        )
        self._kv_write_cache[cache_key] = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Override: attention weight loads (decode path) → HBM bandwidth model
    # ──────────────────────────────────────────────────────────────────────

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        if not self._has_decode(batch):
            return super()._get_attention_layer_pre_proj_execution_time(batch)
        return self._hbm_weight_load_ms(self._qkv_weight_bytes)

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        if not self._has_decode(batch):
            return super()._get_attention_layer_post_proj_execution_time(batch)
        return self._hbm_weight_load_ms(self._o_weight_bytes)

    # ──────────────────────────────────────────────────────────────────────
    # Trace-backed sparse-MoE layer time
    # ──────────────────────────────────────────────────────────────────────

    def _moe_trace_layer_time_ms(self, batch: Batch) -> float:
        decode_pairs = [
            (request, tokens)
            for request, tokens in zip(batch.requests, batch.num_tokens)
            if request.is_prefill_complete
        ]
        if not decode_pairs or self._moe_trace_store is None:
            return 0.0

        request_ids = []
        decode_steps = []
        for request, tokens in decode_pairs:
            for offset in range(max(1, int(tokens))):
                request_ids.append(request.id)
                decode_steps.append(request.num_processed_decode_tokens + offset)
        active_tokens = len(request_ids)
        if active_tokens == 0:
            return 0.0

        M = self._moe_tp_size
        A = self._replica_config.tensor_parallel_size
        D = self._model_config.embedding_dim
        routed_expert_bytes = moe_flash.per_expert_bytes(
            D, self._moe_intermediate, self._moe_dtype_bytes
        )
        shared_expert_bytes = moe_flash.per_expert_bytes(
            D, self._moe_shared_intermediate, self._moe_dtype_bytes
        )
        read_bw_bpns = (
            self._hbm_bandwidth_bpns
            if self._moe_weights_in_hbm
            else self._moe_trace_read_bw_bpns
        )
        routed_fetch_ms = (routed_expert_bytes / read_bw_bpns) / 1e6
        routed_compute_ms = (
            moe_flash.expert_flops_per_token(D, self._moe_intermediate)
            / self._moe_flops_per_ms
        )
        shared_fetch_ms = (shared_expert_bytes / read_bw_bpns) / 1e6
        shared_compute_ms = (
            moe_flash.expert_flops_per_token(D, self._moe_shared_intermediate)
            / self._moe_flops_per_ms
            if self._moe_shared_intermediate > 0
            else 0.0
        )

        policy = self._moe_trace_policy
        hybrid_hbf_gpus = 0
        hybrid_hbm_gpus = 0
        if policy == "hybrid_hbf_dynamic":
            hybrid_hbf_gpus = min(max(int(self._moe_trace_hybrid_hbf_gpus), 0), M)
            hybrid_hbm_gpus = M - hybrid_hbf_gpus
            if hybrid_hbf_gpus <= 0:
                raise ValueError(
                    "moe_trace_policy='hybrid_hbf_dynamic' requires "
                    "moe_trace_hybrid_hbf_gpus > 0"
                )

        previous_key = (policy, M, hybrid_hbf_gpus)
        if policy == "static":
            previous_layers = None
        else:
            previous_layers = self._moe_trace_previous.setdefault(
                previous_key, np.full((TRACE_LAYERS, TRACE_EXPERTS), -1, np.int16)
            )

        layer_times = []
        active_experts = []
        hottest = []
        fetches = []
        hbf_fetch_fractions = []
        hbf_load_fractions = []
        churns = []
        exposed_times = []
        num_layers = min(self._num_layers_per_pipeline_stage, TRACE_LAYERS)
        static_owners = self._moe_trace_store.owners(M) if policy == "static" else None
        hybrid_owners = (
            self._moe_trace_store.limited_owners(
                hybrid_hbm_gpus, self._moe_trace_hybrid_hbm_experts_per_gpu
            )
            if policy == "hybrid_hbf_dynamic"
            else None
        )

        for layer in range(num_layers):
            counts = self._moe_trace_store.counts_for(request_ids, decode_steps, layer)
            active = np.flatnonzero(counts)
            if active.size == 0:
                layer_times.append(0.0)
                exposed_times.append(0.0)
                continue

            previous = None if previous_layers is None else previous_layers[layer]
            if policy == "hybrid_hbf_dynamic":
                owners, churn = assign_hybrid_experts(
                    counts=counts,
                    moe_gpus=M,
                    hbf_gpus=hybrid_hbf_gpus,
                    hbm_static_owner=hybrid_owners[layer],
                    previous=previous,
                    cost_per_expert=routed_fetch_ms,
                    cost_per_token=routed_compute_ms,
                )
            else:
                owners, churn = assign_experts(
                    counts=counts,
                    moe_gpus=M,
                    policy=policy,
                    static_owner=None if static_owners is None else static_owners[layer],
                    previous=previous,
                    cost_per_expert=routed_fetch_ms,
                    cost_per_token=routed_compute_ms,
                )
            if previous_layers is not None:
                previous_layers[layer] = owners
            if churn is not None:
                churns.append(churn)

            assigned = owners[active]
            token_loads = np.bincount(
                assigned, weights=counts[active], minlength=M
            ).astype(float)
            expert_fetches = np.bincount(assigned, minlength=M).astype(float)
            routed_flash = expert_fetches * routed_fetch_ms
            routed_compute = token_loads * routed_compute_ms
            gpu_work = routed_flash + routed_compute
            if policy == "hybrid_hbf_dynamic":
                hbf_work = float(gpu_work[hybrid_hbm_gpus:].sum())
                total_work = float(gpu_work.sum())
                hbf_load_fractions.append(hbf_work / total_work if total_work else 0.0)
                total_fetches = float(expert_fetches.sum())
                hbf_fetch_fractions.append(
                    float(expert_fetches[hybrid_hbm_gpus:].sum()) / total_fetches
                    if total_fetches
                    else 0.0
                )

            shared_flash = shared_fetch_ms if self._moe_shared_intermediate > 0 else 0.0
            shared_compute = (active_tokens / M) * shared_compute_ms
            per_gpu = np.array(
                [
                    moe_flash.combine(
                        float(routed_flash[gpu] + shared_flash),
                        float(routed_compute[gpu] + shared_compute),
                        self._moe_overlap,
                    )
                    for gpu in range(M)
                ]
            )
            moe_ms = float(per_gpu.max())

            if self._disagg_moe_gpus > 0:
                owner_copies = 0
                for request_id, decode_step in zip(request_ids, decode_steps):
                    token_experts = self._moe_trace_store.selected_for(
                        request_id, decode_step, layer
                    )
                    token_owners = owners[token_experts]
                    owner_copies += len(set(int(x) for x in token_owners if x >= 0))
                xfer_bytes = 2 * owner_copies * D * 2
                moe_ms += (xfer_bytes / (min(A, M) * self._disagg_link_bw)) / 1e6

                if self._disagg_overlap:
                    attn_ms = (
                        self._get_attention_layer_pre_proj_execution_time(batch)
                        + self._get_attention_decode_execution_time(batch)
                        + self._get_attention_prefill_execution_time(batch)
                        + self._get_attention_layer_post_proj_execution_time(batch)
                        + self._get_attention_kv_cache_save_execution_time(batch)
                    )
                    moe_ms = max(moe_ms - attn_ms, 0.0)

            layer_times.append(moe_ms)
            exposed_times.append(moe_ms)
            active_experts.append(int(active.size))
            hottest.append(int(counts[active].max()))
            fetches.append(float(expert_fetches.max()))

        result = float(np.mean(layer_times)) if layer_times else 0.0
        result *= num_layers / self._num_layers_per_pipeline_stage
        if self._moe_trace_diagnostics_path is not None:
            record = {
                "batch_id": batch.id,
                "policy": policy,
                "moe_gpus": M,
                "hybrid_hbf_gpus": hybrid_hbf_gpus,
                "hybrid_hbm_gpus": hybrid_hbm_gpus,
                "decode_batch_size": active_tokens,
                "decode_step_min": int(min(decode_steps)),
                "decode_step_max": int(max(decode_steps)),
                "moe_layer_time_ms_mean": result,
                "moe_layer_time_ms_p95": float(np.percentile(layer_times, 95))
                if layer_times
                else 0.0,
                "active_experts_mean": float(np.mean(active_experts))
                if active_experts
                else 0.0,
                "hottest_expert_tokens_p95": float(np.percentile(hottest, 95))
                if hottest
                else 0.0,
                "max_fetched_experts_p95": float(np.percentile(fetches, 95))
                if fetches
                else 0.0,
                "hbf_fetch_fraction_mean": float(np.mean(hbf_fetch_fractions))
                if hbf_fetch_fractions
                else 0.0,
                "hbf_load_fraction_mean": float(np.mean(hbf_load_fractions))
                if hbf_load_fractions
                else 0.0,
                "reassignment_fraction_mean": float(np.mean(churns))
                if churns
                else 0.0,
            }
            with self._moe_trace_diagnostics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Sparse-MoE layer time: expert weights streamed from HBF flash
    # ──────────────────────────────────────────────────────────────────────

    def _moe_layer_time_ms(self, batch: Batch) -> float:
        """
        One MoE layer at this batch: flash expert-weight load ⊕ expert GEMM,
        combined serially (default) or overlapped, exactly as in
        attn_moe_batch_sweep.py.

        Routing volume uses the total tokens in the batch — every token
        (decode or prefill chunk) is routed to top-k experts, so the number
        of unique experts whose weights must be fetched depends on all of
        them. For a pure-decode batch this equals the number of decode
        sequences, matching the sweep script's B.
        """
        if self._moe_trace_store is not None and self._has_decode(batch):
            return self._moe_trace_layer_time_ms(batch)

        num_tokens = sum(batch.num_tokens)
        if num_tokens <= 0:
            return 0.0

        if self._disagg_replicated and not self._disagg_affinity:
            # Balanced routing: tokens load-balance 1/M per GPU, each
            # streaming E_unique(B/M) from its OWN flash stack — no
            # cross-GPU dedup, but per-GPU routing sits lower on the
            # concave E_unique curve. (Affinity routing instead assigns
            # each expert to one GPU per step, restoring the /M dedup of
            # the sharded path below.)
            tokens_per_gpu = num_tokens / self._moe_tp_size
            flash_bytes = moe_flash.moe_flash_bytes(
                B=tokens_per_gpu,
                E=self._moe_num_experts,
                k=self._moe_top_k,
                D=self._model_config.embedding_dim,
                He=self._moe_intermediate,
                be=self._moe_dtype_bytes,
                shared_He=self._moe_shared_intermediate,
                worst=self._moe_routing_worst,
            )
            if self._disagg_clustered:
                # Expert-set-similarity clustering shrinks the routed union
                # per GPU by the efficiency factor; the always-on shared
                # expert is unaffected.
                shared_bytes = moe_flash.per_expert_bytes(
                    self._model_config.embedding_dim,
                    self._moe_shared_intermediate,
                    self._moe_dtype_bytes,
                )
                flash_bytes = (
                    self._disagg_cluster_eff * (flash_bytes - shared_bytes)
                    + shared_bytes
                )
            compute_ms = moe_flash.moe_compute_ms(
                B=tokens_per_gpu,
                k=self._moe_top_k,
                D=self._model_config.embedding_dim,
                He=self._moe_intermediate,
                flops_per_ms=self._moe_flops_per_ms,
                shared_He=self._moe_shared_intermediate,
            )
        else:
            flash_bytes = moe_flash.moe_flash_bytes(
                B=num_tokens,
                E=self._moe_num_experts,
                k=self._moe_top_k,
                D=self._model_config.embedding_dim,
                He=self._moe_intermediate,
                be=self._moe_dtype_bytes,
                shared_He=self._moe_shared_intermediate,
                worst=self._moe_routing_worst,
            ) / self._moe_tp_size
            compute_ms = moe_flash.moe_compute_ms(
                B=num_tokens,
                k=self._moe_top_k,
                D=self._model_config.embedding_dim,
                He=self._moe_intermediate,
                flops_per_ms=self._moe_flops_per_ms,
                shared_He=self._moe_shared_intermediate,
            ) / self._moe_tp_size

        if self._moe_weights_in_hbm:
            # Experts HBM-resident: activated weights stream at HBM bandwidth
            flash_ms = self._hbm_weight_load_ms(flash_bytes)
        elif self._moe_use_sched:
            flash_ms = moe_flash.flash_time_sched_ms(self._hbf, flash_bytes)
        else:
            flash_ms = moe_flash.flash_time_bw_ms(self._hbf, flash_bytes)

        moe_ms = moe_flash.combine(flash_ms, compute_ms, self._moe_overlap)

        # Disaggregated attention/MoE: attention pool (A GPUs) ships every
        # token's hidden state (D x fp16) to the MoE pool (M GPUs) and gets
        # the expert outputs back — serial ping-pong, no micro-batching.
        # A token's top-k experts span ~M x (1 - (1 - 1/M)^k) distinct MoE
        # GPUs, so each token crosses the link that many times per direction.
        # Wire time is bottlenecked by the thinner pool's aggregate links.
        if self._disagg_moe_gpus > 0:
            A = self._replica_config.tensor_parallel_size
            M = self._disagg_moe_gpus
            D = self._model_config.embedding_dim
            if self._disagg_replicated and not self._disagg_affinity:
                # All experts on every MoE GPU: a token visits exactly one,
                # picked by the load balancer.
                copies = 1.0
            else:
                copies = (
                    M * (1.0 - (1.0 - 1.0 / M) ** self._moe_top_k) if M > 1 else 1.0
                )
            xfer_bytes = 2 * num_tokens * copies * D * 2  # dispatch + combine
            moe_ms += (xfer_bytes / (min(A, M) * self._disagg_link_bw)) / 1e6

            # 2-way micro-batch ping-pong: while this micro-batch's MoE runs
            # on the MoE pool, the other micro-batch's attention runs on the
            # attention pool — only MoE time exceeding the attention-side
            # layer time is exposed. (Assumes per-layer expert staging in
            # MoE-GPU HBM so the second micro-batch does not re-stream from
            # flash; all attention hooks below are memoized/analytic, so the
            # extra calls are cheap and idempotent.)
            if self._disagg_overlap:
                attn_ms = (
                    self._get_attention_layer_pre_proj_execution_time(batch)
                    + self._get_attention_decode_execution_time(batch)
                    + self._get_attention_prefill_execution_time(batch)
                    + self._get_attention_layer_post_proj_execution_time(batch)
                    + self._get_attention_kv_cache_save_execution_time(batch)
                )
                moe_ms = max(moe_ms - attn_ms, 0.0)
            return moe_ms

        # Expert-parallel all-to-all (dispatch + combine): each worker forwards
        # its B/TP token shard's activations (D x fp16) to the owners of the
        # top-k experts and gathers the results back; (TP-1)/TP of the traffic
        # crosses the interconnect. Serial with the MoE phase (conservative).
        if self._moe_a2a_bw_gbps > 0 and self._moe_tp_size > 1:
            tp = self._moe_tp_size
            a2a_bytes = (
                2  # dispatch + combine
                * (num_tokens / tp)
                * self._moe_top_k
                * self._model_config.embedding_dim
                * 2  # fp16 activations
                * (tp - 1) / tp
            )
            # bw GB/s = bytes/ns; bytes / (bytes/ns) = ns -> ms
            moe_ms += (a2a_bytes / self._moe_a2a_bw_gbps) / 1e6

        return moe_ms

    # ──────────────────────────────────────────────────────────────────────
    # Override: MLP weight loads (decode path) → HBM bandwidth model,
    # or sparse-MoE expert loads → HBF flash (num_experts > 1)
    # ──────────────────────────────────────────────────────────────────────

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        if self._moe_enabled:
            # Entire MoE layer (gate+up+down flash load + expert GEMM) is
            # accounted here; the down_proj hook returns 0.
            return self._moe_layer_time_ms(batch)
        if not self._has_decode(batch):
            return super()._get_mlp_layer_up_proj_execution_time(batch)
        return self._hbm_weight_load_ms(self._mlp_up_weight_bytes)

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        if self._moe_enabled:
            return 0.0
        if not self._has_decode(batch):
            return super()._get_mlp_layer_down_proj_execution_time(batch)
        return self._hbm_weight_load_ms(self._mlp_down_weight_bytes)

    # ──────────────────────────────────────────────────────────────────────
    # Override: KV cache write → HBM bank scheduler (decode) or bandwidth (prefill)
    # ──────────────────────────────────────────────────────────────────────

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        decode_reqs = self._decode_reqs(batch)
        if decode_reqs:
            return self._hbm_kv_write_ms(
                decode_reqs, self._num_layers_per_pipeline_stage
            )
        # Prefill: large sequential KV write, bandwidth-limited
        num_tokens = sum(batch.num_tokens)
        return (num_tokens * self._kv_write_bytes / self._hbm_bandwidth_bpns) / 1e6

    def _flash_plane_read_ms(self, pages_per_plane: int) -> float:
        """Uniform per-plane read via the NAND plane scheduler, memoized —
        the scheduler drains to a clean state each call, so latency is a
        pure function of pages_per_plane."""
        cached = self._flash_read_cache.get(pages_per_plane)
        if cached is not None:
            return cached
        total_planes = self._hbf._mapper.total_planes
        plane_pages = {p: pages_per_plane for p in range(total_planes)}
        self._hbf.submit_plane_reads(plane_pages, token_id=0)
        completions = self._hbf.drain()
        result = (
            max(c.latency_ns for c in completions) / 1e6 if completions else 0.0
        )
        self._flash_read_cache[pages_per_plane] = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Override: decode KV read → explicit HBF/HBM bandwidth model
    # ──────────────────────────────────────────────────────────────────────

    def _effective_kv_blocks_to_read(self, dense_blocks: int, sparsity: float) -> int:
        """Return the number of KV blocks charged to attention decode.

        Dense attention reads every KV block. Query-dependent top-k sparsity
        reads exactly ceil(sparsity * N) blocks. Naive/random sparsity keeps
        the old "read until the last useful block" expectation, but charges the
        resulting bytes through the bandwidth model instead of the HBF page
        buffer shortcut.
        """
        if dense_blocks <= 0 or sparsity <= 0.0:
            return 0
        if sparsity >= 1.0:
            return dense_blocks
        if self._config.naive_sparse:
            k = max(1.0, sparsity * dense_blocks)
            return min(dense_blocks, math.ceil((dense_blocks - 1) * k / (k + 1)) + 1)
        return max(1, math.ceil(dense_blocks * sparsity))

    def _hbf_kv_read_ms(self, kv_blocks: int) -> float:
        """Time HBF-resident attention KV reads as bytes / configured bandwidth."""
        if kv_blocks <= 0:
            return 0.0
        kv_bytes = kv_blocks * self._kv_block_bytes
        return kv_bytes / (self._hbf_kv_read_bandwidth_bpns * 1e6)

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        """
        Decode attention KV read latency.

        Two modes controlled by hbm_kv_fraction:

        Baseline (hbm_kv_fraction == 0):
          All KV is in HBF. Charge all layers' KV bytes through an explicit
          bandwidth model and return hbf_kv_read_total / num_layers.

        Pipeline (hbm_kv_fraction > 0):
          Hot window (most-recent hbm_kv_fraction of blocks) is read from HBM.
          Cold window is read from HBF. The exposed per-layer stall is
          max(T_hbm, T_hbf) / num_layers.
        """
        decode_reqs = self._decode_reqs(batch)
        if not decode_reqs:
            return 0.0

        num_layers = self._num_layers_per_pipeline_stage
        sparsity = self._config.sparsity_fraction
        hbm_kv_fraction = self._config.hbm_kv_fraction

        cached = self._config.cached_context_tokens
        total_kv_blocks_per_layer = sum(
            max(1, math.ceil((r.num_processed_tokens + cached) / self._block_size))
            for r in decode_reqs
        )

        if hbm_kv_fraction > 0.0:
            hbm_blocks = max(1, round(total_kv_blocks_per_layer * hbm_kv_fraction))
            hbm_blocks = min(hbm_blocks, total_kv_blocks_per_layer)
            hbf_blocks = max(0, total_kv_blocks_per_layer - hbm_blocks)

            hbm_kv_bytes = hbm_blocks * num_layers * self._kv_block_bytes
            hbm_time_ms = hbm_kv_bytes / (self._hbm_bandwidth_bpns * 1e6)

            hbf_blocks_to_read = self._effective_kv_blocks_to_read(
                hbf_blocks * num_layers,
                sparsity,
            )
            hbf_time_ms = self._hbf_kv_read_ms(hbf_blocks_to_read)

            return max(hbm_time_ms, hbf_time_ms) / num_layers

        total_dense_blocks = total_kv_blocks_per_layer * num_layers
        hbf_blocks_to_read = self._effective_kv_blocks_to_read(
            total_dense_blocks,
            sparsity,
        )
        return self._hbf_kv_read_ms(hbf_blocks_to_read) / num_layers

    # ──────────────────────────────────────────────────────────────────────
    # Step boundary: reset HBM clock before each batch
    # ──────────────────────────────────────────────────────────────────────

    def get_execution_time(self, batch: Batch, pipeline_stage: int):
        # Each batch is an independent decode step; Vidur's event loop handles
        # inter-batch timing.  Reset so each _get_* call measures from t=0.
        self._hbm.reset_step()
        return super().get_execution_time(batch, pipeline_stage)

    # ──────────────────────────────────────────────────────────────────────
    # Accessors for post-run analysis
    # ──────────────────────────────────────────────────────────────────────

    def memory_backend(self) -> MemoryBackend:
        return self._memory_backend

    def hbf_backend(self) -> HBFSimBackend:
        return self._hbf_backend

    def hbm_backend(self) -> RamulatorBackend:
        return self._hbm
