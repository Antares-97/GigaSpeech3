"""Shim module: re-exports everything from zipformer.utils.

Placed at the asr/ level so that absolute imports like
    from utils import get_avg_checkpoint, LayerNorm, ...
within both the training modules and zipformer/ modules resolve correctly.
The actual implementation lives in zipformer.utils.
"""

from zipformer.utils import *  # noqa: F401, F403
