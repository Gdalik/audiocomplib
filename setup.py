from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extensions
extensions = [
    Extension(
        "audiocomplib.smooth_gain_reduction",
        sources=["audiocomplib/smooth_gain_reduction.pyx"],
        include_dirs=[numpy.get_include()]  # Include NumPy headers
    )
]

# Use cythonize to compile the extensions and let setuptools handle the location
setup(
    name="audiocomplib",
    ext_modules=cythonize(extensions),
    zip_safe=False
)
