"""Shim module: re-exports scaling classes for zipformer compatibility.

Many zipformer/ modules do absolute imports like:
    from scaling import ScaledLinear, Balancer, ...
The actual implementation lives in zipformer.scaling.
"""

from zipformer.scaling import *  # noqa: F401, F403
