from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "audiocomplib.smooth_gain_reduction",
        sources=["audiocomplib/smooth_gain_reduction.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="audiocomplib",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
