"""Torch-native stand-ins for sarathi's tensor-parallel layers.

Used to run Vidur's MLP profiler on GPUs sarathi-serve can't build for (e.g.
Blackwell sm_120). The matmuls are plain nn.Linear -> cuBLAS, i.e. the same GEMM
any framework dispatches, so GEMM timings are faithful on the real GPU. Each
layer times its own matmul under `linear_metric_name` via Vidur's CudaTimer, with
TP modeled by dividing the parallel dimension by world_size (single-GPU shapes,
exactly as Vidur profiles TP on one GPU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vidur.profiling.common.cuda_timer import CudaTimer


class ColumnParallelLinear(nn.Module):
    # splits OUTPUT features across workers
    def __init__(self, input_size, output_size, bias=False, gather_output=False,
                 linear_metric_name="", world_size=1, rank=0, **kw):
        super().__init__()
        assert output_size % world_size == 0
        self.linear = nn.Linear(input_size, output_size // world_size, bias=bias)
        self._timer = CudaTimer(linear_metric_name)

    def forward(self, x):
        with self._timer:
            y = self.linear(x)
        return y, None


class RowParallelLinear(nn.Module):
    # splits INPUT features across workers
    def __init__(self, input_size, output_size, bias=False, input_is_parallel=True,
                 reduce_results=False, linear_metric_name="", world_size=1, rank=0, **kw):
        super().__init__()
        assert input_size % world_size == 0
        self.linear = nn.Linear(input_size // world_size, output_size, bias=bias)
        self._timer = CudaTimer(linear_metric_name)

    def forward(self, x):
        with self._timer:
            y = self.linear(x)
        return y, None


class VocabParallelEmbedding(nn.Module):
    # splits VOCAB across workers
    def __init__(self, num_embeddings, embedding_dim, linear_metric_name="",
                 reduce_results=False, world_size=1, rank=0, **kw):
        super().__init__()
        assert num_embeddings % world_size == 0
        self.emb = nn.Embedding(num_embeddings // world_size, embedding_dim)
        self._timer = CudaTimer(linear_metric_name)

    def forward(self, input_ids):
        with self._timer:
            return self.emb(input_ids)
