from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "audiocomplib",
        sources=["audiocomplib/apply_gain_reduction.pyx"],
        include_dirs=[numpy.get_include()]  # Include NumPy headers
    )
]

setup(
    name="audiocomplib",
    ext_modules=cythonize(extensions),
)
