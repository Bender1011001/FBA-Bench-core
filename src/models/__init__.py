import os
import sys
import warnings

# Add the src directory to Python path so we can import fba_bench_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fba_bench_core.models import *  # noqa: F403
except ImportError:
    # Fallback - define the legacy Product model here instead
    from .product import Product  # noqa: F401

warnings.warn(
    "The 'models' package is deprecated; use 'fba_bench_core.models'. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
