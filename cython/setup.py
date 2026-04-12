"""
Setup script to compile Cython extensions for ZANE.

Compile high-performance fingerprint operations into C extensions.
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="zane.fingerprints",
        sources=["cython/fingerprints.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
        extra_compile_args=["-O3", "-march=native"],
        parallel=True,  # OpenMP parallelization
    )
]

setup(
    name="zane_cython_extensions",
    version="0.1.0",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        nthreads=4,
    ),
    include_dirs=[numpy.get_include()],
)
