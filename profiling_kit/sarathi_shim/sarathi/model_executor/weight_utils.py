import torch


def initialize_dummy_weights(model, low: float = -1e-3, high: float = 1e-3):
    """Fill all params with small random values (no real weights needed for
    timing). Mirrors sarathi.model_executor.weight_utils.initialize_dummy_weights."""
    for p in model.state_dict().values():
        if torch.is_floating_point(p):
            p.data.uniform_(low, high)
