"""
HBF-backed linear-regression execution time predictor.

All memory-bound operations are replaced by cycle-accurate simulation:

  HBF plane scheduler  →  cold KV cache reads (decode)
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

import math

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
from vidur.memory_backends import (
    CompositeBackend,
    HBFSimBackend,
    PlacementPolicy,
    RamulatorBackend,
)
from vidur.memory_backends.base import MemoryBackend, OpType
from vidur.memory_backends.hbf.config_loader import HBFSimConfigLoader


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
      _get_attention_decode_execution_time           KV reads         → HBF plane sched

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
        # HBM bandwidth must be DEVICE-SPECIFIC: long-context decode is memory
        # bound, so weight loads / KV writes / HBM hot-window reads scale with
        # the target GPU's HBM bandwidth — not a fixed config value. Using the
        # TOML's single bandwidth makes A40 and H100 decode identical (the F6
        # device-invariance bug). Prefer the device SKU; fall back to the TOML.
        # bandwidth is GB/s; numerically 1 GB/s = 1 byte/ns.
        device_bw = getattr(replica_config.device_config, "mem_bandwidth_gbps", None)
        self._hbm_bandwidth_bpns = float(device_bw) if device_bw else hbf_cfg.hbm.bandwidth_gbps

        # ── Model geometry ────────────────────────────────────────────────
        E   = self._model_config.embedding_dim
        H   = self._model_config.mlp_hidden_dim
        Nq  = self._model_config.num_q_heads
        Nkv = self._model_config.num_kv_heads
        D   = E // Nq  # head_dim

        # ── Weight sizes per layer (FP16 = 2 bytes) ───────────────────────
        # QKV: Q (E×NqD) + K (E×NkvD) + V (E×NkvD)
        self._qkv_weight_bytes = (E * Nq * D + 2 * E * Nkv * D) * 2
        # O projection: NqD × E
        self._o_weight_bytes = Nq * D * E * 2
        # MLP gate fused with up in gated architectures (LLaMA SwiGLU)
        _gate = 2 if self._model_config.use_gated_mlp else 1
        self._mlp_up_weight_bytes   = E * H * _gate * 2
        self._mlp_down_weight_bytes = H * E * 2

        # ── KV sizes ──────────────────────────────────────────────────────
        # Full KV block: K+V, block_size tokens, FP16
        self._kv_block_bytes = 2 * self._block_size * Nkv * D * 2
        # One new token write per decode step: K+V, 1 token, FP16
        self._kv_write_bytes = 2 * Nkv * D * 2

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

        self._req_counter = 0

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
        if not completions:
            return 0.0
        return (max(c.latency_ns for c in completions) / 1e6) / num_layers

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
    # Override: MLP weight loads (decode path) → HBM bandwidth model
    # ──────────────────────────────────────────────────────────────────────

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        if not self._has_decode(batch):
            return super()._get_mlp_layer_up_proj_execution_time(batch)
        return self._hbm_weight_load_ms(self._mlp_up_weight_bytes)

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
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

    # ──────────────────────────────────────────────────────────────────────
    # Override: decode KV read → HBF plane scheduler (+ HBM for hot tier)
    # ──────────────────────────────────────────────────────────────────────

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        """
        Replace sklearn attn_decode model with cycle-accurate NAND plane simulation.

        Two modes controlled by hbm_kv_fraction:

        Baseline (hbm_kv_fraction == 0):
          All KV in HBF. Sparsity applied per-plane over total blocks.
          Returns: flash_stall_total / num_layers

        Pipeline (hbm_kv_fraction > 0):
          Hot window (most-recent hbm_kv_fraction of blocks) stays in HBM; the
          remaining cold KV has spilled to HBF and is read with sparsity. Within
          a decode step the hot HBM read and the sparse HBF read happen serially
          — the spilled fraction is fetched *in addition to* the hot window — so
          the two times add rather than overlap.
            T_hbm  = num_layers × hbm_window_kv_bytes / hbm_bw  (serial, layer by layer)
            T_flash = plane scheduler latency for all layers' sparse HBF reads
            Effective decode time = T_hbm + T_flash
          Returns: (T_hbm + T_flash) / num_layers

        Efficiency: instead of submitting O(B × L × K) events to the plane scheduler,
        compute the per-plane page load analytically (STRIPE mapping) and submit one
        aggregate request per plane — 768 events instead of ~65 K.
        """
        decode_reqs = self._decode_reqs(batch)
        if not decode_reqs:
            return 0.0

        num_layers        = self._num_layers_per_pipeline_stage
        sparsity          = self._config.sparsity_fraction
        hbm_kv_fraction   = self._config.hbm_kv_fraction
        page_size         = self._hbf._cfg.subarray.page_size_bytes
        total_planes      = self._hbf._mapper.total_planes

        total_kv_blocks_per_layer = sum(
            max(1, math.ceil(r.num_processed_tokens / self._block_size))
            for r in decode_reqs
        )

        if hbm_kv_fraction > 0.0:
            # ── Pipeline: HBM hot window + HBF sparse cold window ────────────
            hbm_blocks = max(1, round(total_kv_blocks_per_layer * hbm_kv_fraction))
            hbf_blocks = max(0, total_kv_blocks_per_layer - hbm_blocks)

            # HBM time: read all layers' hot KV sequentially (layer-serial)
            hbm_kv_bytes = hbm_blocks * num_layers * self._kv_block_bytes
            hbm_time_ms  = hbm_kv_bytes / (self._hbm_bandwidth_bpns * 1e6)

            # HBF time: sparse on cold portion, all layers staged simultaneously
            if hbf_blocks > 0 and sparsity > 0:
                total_hbf_blocks = hbf_blocks * num_layers
                dense_pp  = math.ceil(total_hbf_blocks / total_planes)
                sparse_pp = max(1, math.ceil(dense_pp * sparsity))
                pages_per_plane = max(1, math.ceil(sparse_pp * self._kv_block_bytes / page_size))
                plane_pages  = {p: pages_per_plane for p in range(total_planes)}
                plane_blocks = {p: sparse_pp for p in range(total_planes)}
                self._hbf.submit_plane_reads(
                    plane_pages, token_id=0, plane_block_counts=plane_blocks)
                completions = self._hbf.drain()
                flash_time_ms = (max(c.latency_ns for c in completions) / 1e6
                                 if completions else 0.0)
            else:
                flash_time_ms = 0.0

            # HBM hot window and HBF cold window are read serially within the
            # decode step: the spilled KV is fetched from HBF (sparse fraction)
            # in addition to the hot KV from HBM, so the times add.
            return (hbm_time_ms + flash_time_ms) / num_layers

        else:
            # ── Baseline: all KV in HBF ───────────────────────────────────────
            total_dense_blocks     = total_kv_blocks_per_layer * num_layers
            blocks_per_plane_dense = math.ceil(total_dense_blocks / total_planes)

            if self._config.naive_sparse and sparsity < 1.0:
                # Naive (random) sparsity: sparsity_fraction of blocks are useful but
                # their positions are unknown, so we read sequentially until the last
                # useful block is encountered.
                # E[last useful block] = ceil((N-1) * k/(k+1)) + 1
                # where k = number of useful blocks per plane = sparsity * N.
                # For small sparsity this ≈ N (nearly dense), showing random sparsity
                # provides almost no bandwidth saving without selective access.
                n = blocks_per_plane_dense
                k = max(1, sparsity * n)
                blocks_per_plane_sparse = math.ceil((n - 1) * k / (k + 1)) + 1
            else:
                # Top-K selection (your model): each sequence reads its k most
                # relevant blocks per layer, k = ceil(seq_blocks * sparsity).
                # Apply the selection per request *before* striping across planes
                # so the per-plane count matches the physical access pattern (the
                # trace-level sim); rounding sparsity after the divide-by-planes
                # would drift by up to one block per plane.
                total_selected_per_layer = sum(
                    max(1, math.ceil(
                        max(1, math.ceil(r.num_processed_tokens / self._block_size))
                        * sparsity))
                    for r in decode_reqs
                )
                total_selected_blocks = total_selected_per_layer * num_layers
                blocks_per_plane_sparse = max(
                    1, math.ceil(total_selected_blocks / total_planes))

            pages_per_plane = max(1, math.ceil(blocks_per_plane_sparse * self._kv_block_bytes / page_size))
            plane_pages  = {p: pages_per_plane for p in range(total_planes)}
            plane_blocks = {p: blocks_per_plane_sparse for p in range(total_planes)}
            self._hbf.submit_plane_reads(
                plane_pages, token_id=0, plane_block_counts=plane_blocks)
            completions = self._hbf.drain()
            if not completions:
                return 0.0
            return (max(c.latency_ns for c in completions) / 1e6) / num_layers

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
