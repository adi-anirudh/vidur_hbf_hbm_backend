import json
import os
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.config.utils import dataclass_to_dict
from vidur.logger import init_logger
from vidur.types import (
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    def __post_init__(self):
        self.max_tokens = self.prefill_tokens + self.decode_tokens

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )
    decode_only: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, requests start with prefill already complete. "
                "num_processed_tokens is initialised to num_prefill_tokens so the "
                "request enters decode immediately, bypassing the prefill queue. "
                "Use with --sarathi_scheduler_config_num_blocks to override the "
                "GPU-memory-based block limit when modelling an HBF system where "
                "KV lives in flash, not GPU SRAM."
            )
        },
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE_REPLAY


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    batch_size_cap: int = field(
        default=128,
        metadata={"help": "Maximum batch size cap."},
    )
    block_size: int = field(
        default=16,
        metadata={"help": "Block size."},
    )
    watermark_blocks_fraction: float = field(
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    num_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks."},
    )


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for vLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class LightllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for LightLLM."},
    )
    max_waiting_iters: int = field(
        default=10,
        metadata={"help": "Maximum waiting iterations for LightLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.LIGHTLLM


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )
    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="simulator_output",
        metadata={"help": "Output directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ReplicaConfig:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name."},
    )
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )

    def __post_init__(self):
        self.world_size = self.num_pipeline_stages * self.tensor_parallel_size
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN


@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    compute_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv",
        metadata={"help": "Path to the compute input file."},
    )
    attention_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class HBFLinearRegressionExecutionTimePredictorConfig(
    LinearRegressionExecutionTimePredictorConfig
):
    hbfsim_config_path: str = field(
        default="",
        metadata={"help": "Path to the HBFSim TOML config for NAND plane parameters."},
    )
    placement_policy: str = field(
        default="STRIPE_ACROSS_PLANES",
        metadata={
            "help": (
                "KV block placement policy across NAND planes. "
                "One of: STRIPE_ACROSS_PLANES, PACK_BY_SEQUENCE, INTERLEAVE_BY_TOKEN."
            )
        },
    )
    hot_kv_fraction: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of decode sequences whose KV is in HBM (hot tier). "
                "Remainder is fetched from HBF (cold tier). "
                "Requires use_ramulator=True for non-zero values."
            )
        },
    )
    use_ramulator: bool = field(
        default=False,
        metadata={"help": "Also simulate the HBM tier with a row-buffer bank scheduler."},
    )
    sparsity_fraction: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fraction of KV blocks to read per (sequence, layer) at each decode step. "
                "1.0 = dense attention (all blocks). 0.1 = 10%% sparsity (top-K' selection). "
                "Models query-dependent sparsity where only the most relevant KV blocks are read."
            )
        },
    )
    naive_sparse: bool = field(
        default=False,
        metadata={
            "help": (
                "Model naive (random) sparsity: sparsity_fraction of blocks are marked useful "
                "but their positions are unknown ahead of time, so blocks are read sequentially "
                "until the last useful one is encountered. "
                "Expected reads = ceil((N-1) * k/(k+1)) + 1 ≈ N for small sparsity, "
                "showing that random sparsity provides almost no bandwidth saving vs dense. "
                "Contrast with sparsity_fraction alone (top-K selection), which reads exactly k blocks."
            )
        },
    )
    cached_context_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Prefix-cache emulation: every request is assumed to carry "
                "this many context tokens whose KV is ALREADY resident (from "
                "a cached prefix, e.g. multi-turn history) — no prefill is "
                "paid for them, but decode KV reads cover them every step. "
                "Pair with a small prefill_tokens (the new turn)."
            )
        },
    )
    assume_dp_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Colocated baseline correction for MLA models: treat the "
                "replica as DP-attention + EP-experts (SGLang/DeepSeek-style) "
                "— per-worker KV/weight/expert byte accounting is unchanged, "
                "but the per-layer tensor-parallel all-reduce is dropped "
                "(DP attention has none; experts pay only the modeled "
                "all-to-all)."
            )
        },
    )
    tp_shard_scope: str = field(
        default="experts",
        metadata={
            "help": (
                "What tensor parallelism shards across TP workers. "
                "'experts' (default): only MoE expert weights/GEMM divide by TP; "
                "attention weights and KV cache are modeled at full single-GPU "
                "cost (original HBF-study behavior). "
                "'full': attention weights, dense MLP weights, and KV cache are "
                "also sharded 1/TP — modeling a conventional TP deployment where "
                "the workers jointly hold the model and KV within their "
                "per-device memory limits."
            )
        },
    )
    enforce_memory_check: bool = field(
        default=True,
        metadata={
            "help": (
                "Validate at startup that per-GPU resident bytes fit device "
                "capacity: HBM-resident weights vs total_memory_gb, and "
                "HBF-resident expert weights vs flash stack capacity. "
                "Fails fast with a capacity report instead of silently "
                "simulating an infeasible placement."
            )
        },
    )
    hbm_kv_fraction: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of each sequence's most-recent KV blocks stored in HBM (hot window). "
                "The remaining (1 - hbm_kv_fraction) cold blocks are fetched from HBF. "
                "0.0 = all KV in HBF (baseline). 0.0909 ≈ 1:10 HBM:HBF split. "
                "When > 0, the predictor uses a pipeline model where HBM dense attention "
                "and HBF sparse reads execute concurrently; effective decode stall = "
                "max(hbm_dense_attention_time, hbf_sparse_read_time)."
            )
        },
    )
    hbf_kv_read_bw_gbps: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bandwidth in GB/s used for HBF-resident attention KV reads. "
                "0.0 means use the configured HBM bandwidth, matching the "
                "paper model where HBM and HBF read bandwidth are equal. "
                "This affects attention/KV timing only, not MoE expert fetches."
            )
        },
    )
    # ── Sparse-MoE with expert weights on HBF flash ───────────────────────
    # Mirrors attn_moe_batch_sweep.py; math shared via vidur.memory_backends.moe_flash.
    num_experts: int = field(
        default=1,
        metadata={
            "help": (
                "Total routed experts E per MoE layer. 1 = dense MLP (baseline "
                "behavior: weights streamed from HBM, unchanged). > 1 enables the "
                "MoE-on-HBF decode path: activated expert weights are streamed "
                "from HBF flash each decode step."
            )
        },
    )
    moe_top_k: int = field(
        default=1,
        metadata={"help": "Experts activated per token (top-k routing)."},
    )
    moe_intermediate: int = field(
        default=0,
        metadata={
            "help": (
                "Per-expert hidden dim H_e. 0 = fall back to the model config's "
                "mlp_hidden_dim."
            )
        },
    )
    moe_dtype_bytes: int = field(
        default=2,
        metadata={"help": "Expert weight bytes on flash (fp16=2, fp8=1)."},
    )
    shared_expert_intermediate: int = field(
        default=0,
        metadata={
            "help": (
                "Intermediate size of an always-on shared expert (DeepSeek-style); "
                "flat flash+compute floor, not routed. 0 = none (Qwen)."
            )
        },
    )
    moe_overlap: bool = field(
        default=False,
        metadata={
            "help": (
                "If False (default), MoE layer time = flash load + expert GEMM "
                "(serial). If True, max(flash, compute) (perfect overlap)."
            )
        },
    )
    moe_routing: str = field(
        default="uniform",
        metadata={
            "help": (
                "Expert-routing model for unique-expert count: 'uniform' "
                "(expected distinct experts under uniform-random routing) or "
                "'worst' (min(B*k, E))."
            )
        },
    )
    moe_trace_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional R1 no-pruning replay directory. When set, the MoE "
                "decode path uses held-out selected-expert traces instead of "
                "the uniform/worst synthetic routing model."
            )
        },
    )
    moe_trace_policy: str = field(
        default="token_count_dynamic",
        metadata={
            "help": (
                "Trace-backed expert assignment policy: static, "
                "active_count_dynamic, token_count_dynamic, cost_dynamic, "
                "or hybrid_hbf_dynamic."
            )
        },
    )
    moe_trace_workload: str = field(
        default="mixed",
        metadata={
            "help": (
                "Held-out trace workload to replay from moe_trace_path; use "
                "'mixed' to cycle across all held-out requests."
            )
        },
    )
    moe_trace_max_requests: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum held-out trace requests to load into the Vidur "
                "trace-backed MoE route pool."
            )
        },
    )
    moe_trace_profile_max_requests: int = field(
        default=0,
        metadata={
            "help": (
                "Maximum train requests to use for static placement profiling. "
                "0 means all matching train requests."
            )
        },
    )
    moe_trace_read_bw_gbps: float = field(
        default=1024.0,
        metadata={
            "help": (
                "Bandwidth in GB/s used for trace-backed expert-weight fetches. "
                "Default keeps HBM and HBF expert-read bandwidth equal at 1 TB/s."
            )
        },
    )
    moe_trace_hybrid_hbf_gpus: int = field(
        default=0,
        metadata={
            "help": (
                "For moe_trace_policy='hybrid_hbf_dynamic': number of MoE GPUs "
                "that are HBF full-expert balancers. Remaining MoE GPUs are HBM "
                "capacity-limited static owners."
            )
        },
    )
    moe_trace_hybrid_hbm_experts_per_gpu: int = field(
        default=29,
        metadata={
            "help": (
                "For moe_trace_policy='hybrid_hbf_dynamic': routed expert slots "
                "per HBM MoE GPU per layer after reserving shared expert space. "
                "Default 29 matches the 72 GiB HBM / FP8 R1-V3 model."
            )
        },
    )
    moe_trace_diagnostics_path: str = field(
        default="",
        metadata={
            "help": (
                "Optional JSONL path for per-Vidur-batch trace-backed MoE "
                "diagnostics."
            )
        },
    )
    moe_weights_in_hbm: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, expert weights reside in HBM instead of HBF flash: "
                "activated expert loads are timed at HBM bandwidth and expert "
                "bytes count against the per-GPU HBM budget in the memory "
                "check (conventional all-HBM baseline). Combine with "
                "tp_shard_scope='full' and hbm_kv_fraction=1.0 for a fully "
                "HBM-resident TP deployment."
            )
        },
    )
    moe_a2a_bw_gbps: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-GPU interconnect bandwidth (GB/s) for MoE expert-parallel "
                "all-to-all dispatch+combine. 0 (default) = not modeled (free "
                "routing, the original behavior). When > 0 and TP > 1, each MoE "
                "layer adds 2 x (B/TP) x top_k x D x 2B x (TP-1)/TP / bw — "
                "each worker forwards its token shard's activations to expert "
                "owners and gathers results back. H100 NVLink ~450 GB/s "
                "unidirectional per GPU."
            )
        },
    )
    disagg_num_moe_gpus: int = field(
        default=0,
        metadata={
            "help": (
                "Attention/MoE disaggregation: number of dedicated MoE GPUs "
                "holding the expert weights (sharded 1/M on their HBF stacks). "
                "0 (default) = colocated (original behavior). When > 0, run "
                "with tensor_parallel_size = number of ATTENTION GPUs; per "
                "layer, the attention pool computes attention, ships hidden "
                "states to the MoE pool, waits for expert outputs, then starts "
                "the next layer (serial, no micro-batching). Total cluster "
                "GPUs = tensor_parallel_size + disagg_num_moe_gpus."
            )
        },
    )
    disagg_expert_placement: str = field(
        default="sharded",
        metadata={
            "help": (
                "Disaggregated mode. 'sharded': experts split 1/M across MoE "
                "GPUs (Megatron-style; tokens visit the owners of their top-k "
                "experts). 'replicated': every MoE GPU holds ALL experts on "
                "its own HBF stack (needs the large-capacity TOML); tokens are "
                "load-balanced 1-copy across MoE GPUs, each streaming "
                "E_unique(B/M) from its own flash. In replicated mode the "
                "attention pool is sequence-DP: full weight copy per GPU, KV "
                "owned per-sequence (1/A), and NO all-reduce anywhere."
            )
        },
    )
    disagg_routing: str = field(
        default="balanced",
        metadata={
            "help": (
                "Token routing for disagg_expert_placement='replicated'. "
                "'balanced': tokens split 1/M evenly; each MoE GPU streams "
                "E_unique(B/M) redundantly (simple, 1-copy transfers). "
                "'affinity': each expert is assigned to one MoE GPU per step "
                "(possible because every GPU holds all experts); tokens go to "
                "their experts' owners — per-GPU flash drops to "
                "E_unique(B)/M (full dedup) at the cost of multi-copy "
                "transfers, like sharded placement."
            )
        },
    )
    disagg_cluster_efficiency: float = field(
        default=0.75,
        metadata={
            "help": (
                "For disagg_routing='clustered': factor by which expert-set-"
                "similarity clustering shrinks each MoE GPU's expert union "
                "below the random-grouping value E_unique(B/M). 1.0 = no "
                "benefit (= balanced); ~0.7-0.8 is an optimistic bound for "
                "uniform top-k routing (pairwise token overlap is only "
                "k^2/E experts)."
            )
        },
    )
    disagg_microbatch_overlap: bool = field(
        default=False,
        metadata={
            "help": (
                "Disaggregated mode only: 2-way micro-batch ping-pong. While "
                "the MoE pool runs micro-batch 1's experts, the attention pool "
                "runs micro-batch 2's attention; per layer the exposed MoE "
                "time is max(T_moe - T_attn, 0) instead of T_moe. Assumes the "
                "MoE pool stages the current layer's streamed experts in HBM "
                "so the second micro-batch does not re-stream from flash "
                "(buffer: ~E x expert_bytes / M per layer). False (default) = "
                "serial ping-pong."
            )
        },
    )
    disagg_link_bw_gbps: float = field(
        default=450.0,
        metadata={
            "help": (
                "Per-GPU interconnect bandwidth (GB/s) for attention<->MoE "
                "pool transfers in disaggregated mode. Transfer time = total "
                "bytes / (min(A, M) x bw) — the thinner pool's aggregate "
                "links are the bottleneck."
            )
        },
    )
    moe_use_sched: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, time expert flash loads with the cycle-accurate NAND "
                "plane scheduler (optimistic tRC streaming). Default False uses "
                "the conservative peak-bandwidth bound — the sweep script's "
                "headline (its skip_sched fast path)."
            )
        },
    )

    def __post_init__(self):
        assert self.tp_shard_scope in ("experts", "full"), (
            "tp_shard_scope must be 'experts' or 'full'"
        )
        assert self.disagg_expert_placement in ("sharded", "replicated"), (
            "disagg_expert_placement must be 'sharded' or 'replicated'"
        )
        assert self.disagg_routing in ("balanced", "affinity", "clustered"), (
            "disagg_routing must be 'balanced', 'affinity', or 'clustered'"
        )
        assert self.num_experts >= 1, "num_experts must be >= 1"
        if self.num_experts > 1:
            assert 1 <= self.moe_top_k <= self.num_experts, (
                "moe_top_k must be in [1, num_experts]"
            )
            assert self.moe_dtype_bytes >= 1, "moe_dtype_bytes must be >= 1"
            assert self.moe_routing in ("uniform", "worst"), (
                "moe_routing must be 'uniform' or 'worst'"
            )
            assert self.moe_trace_policy in (
                "static",
                "active_count_dynamic",
                "token_count_dynamic",
                "cost_dynamic",
                "hybrid_hbf_dynamic",
            ), "invalid moe_trace_policy"
            assert self.moe_trace_hybrid_hbf_gpus >= 0, (
                "moe_trace_hybrid_hbf_gpus must be non-negative"
            )
            assert self.moe_trace_hybrid_hbm_experts_per_gpu >= 0, (
                "moe_trace_hybrid_hbm_experts_per_gpu must be non-negative"
            )
            assert self.moe_trace_max_requests >= 1, (
                "moe_trace_max_requests must be >= 1"
            )
            assert self.moe_trace_profile_max_requests >= 0, (
                "moe_trace_profile_max_requests must be >= 0"
            )
            assert self.moe_trace_read_bw_gbps > 0, (
                "moe_trace_read_bw_gbps must be positive"
            )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.HBF_LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST


@dataclass
class ClusterConfig:
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},
    )
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig,
        metadata={"help": "Global scheduler config."},
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )


@dataclass
class SimulationConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},
    )

    def __post_init__(self):
        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
