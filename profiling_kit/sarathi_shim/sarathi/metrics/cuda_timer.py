# Placeholder; vidur's MlpWrapper monkey-patches this symbol with its own
# CudaTimer (see vidur/profiling/mlp/mlp_wrapper.py). Only needs to exist so the
# `import sarathi.metrics.cuda_timer` succeeds before the patch.
class CudaTimer:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
