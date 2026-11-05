"""Shim module: re-exports optim classes for finetune_tri_stage compatibility.

The training/decoding modules import `optim` and reference optim.LRScheduler.
The actual implementation lives in zipformer.optim.
"""

# Re-export everything the training/decoding scripts might need
from zipformer.optim import (  # noqa: F401
    BatchedOptimizer,
    Eden,
    Eden2,
    Eve,
    LRScheduler,
    Optimizer,
    ScaledAdam,
)
