import warnings
from .smooth_gain_reduction_py import smooth_gain_reduction as smooth_gain_reduction_py


# Try to import the Cython version
try:
    from .smooth_gain_reduction import smooth_gain_reduction as smooth_gain_reduction_cy
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    warnings.filterwarnings("default", category=ImportWarning)
    warnings.warn("Could not import the 'smooth_gain_reduction' Cython module!\n"
                  "Falling back to pure Python, which may significantly decrease the speed.", category=ImportWarning)

# Use the Cython version if available, otherwise use the Python version
smooth_gain_reduction = smooth_gain_reduction_cy if USE_CYTHON else smooth_gain_reduction_py
