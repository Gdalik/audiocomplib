"""
Smart import handler for smooth_gain_reduction (v0.2.0).

Tries Cython first, falls back to pure Python.
"""

import warnings

from .smooth_gain_reduction_py import smooth_gain_reduction as smooth_gain_reduction_py

try:
    from .smooth_gain_reduction import smooth_gain_reduction as smooth_gain_reduction_cy
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    warnings.warn(
        "Could not import Cython-compiled 'smooth_gain_reduction' module.\n"
        "Using pure Python fallback (significantly slower, ~20x).\n"
        "To enable Cython: pip install -e . --force-reinstall --no-cache-dir",
        category=ImportWarning,
        stacklevel=2
    )

smooth_gain_reduction = smooth_gain_reduction_cy if USE_CYTHON else smooth_gain_reduction_py
