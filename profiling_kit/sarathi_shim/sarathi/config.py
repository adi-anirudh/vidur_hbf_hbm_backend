from dataclasses import dataclass


@dataclass
class ParallelConfig:
    """Minimal stand-in: Vidur's profiler only reads tensor_parallel_size."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
