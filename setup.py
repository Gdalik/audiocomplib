from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define Cython extension for the apply_gain_reduction Cython module
cython_extension = Extension(
    "audiocomplib.apply_gain_reduction",  # Correct module path
    sources=["audiocomplib/apply_gain_reduction.pyx"],
    include_dirs=[np.get_include()],  # Include NumPy headers
)

setup(
    name="audiocomplib",
    version="0.1.0",
    description="A Python library for audio dynamic compression and limiting",
    long_description_content_type="text/markdown",
    long_description=open("README.md", "r").read(),  # Include long description from README
    author="Gdaliy Garmiza",
    author_email="gdalik@gmail.com",
    url="https://github.com/Gdalik/audiocomplib",  # Update with your actual URL
    packages=["audiocomplib"],  # List your package(s) here
    ext_modules=cythonize([cython_extension]),  # Build Cython extension
    install_requires=[
        "numpy",
        "cython",
    ],
    extras_require={
        "dev": ["pytest", "sphinx"],  # Dependencies for development
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",  # Updated to match your python_requires version
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    test_suite="tests",  # Ensure that your tests are in the 'tests' directory
    tests_require=["pytest"],
    zip_safe=False,  # Required due to Cython extensions
)
