from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(
    name="Chi square",
    ext_modules=cythonize("pairwise_fast.pyx"),
    include_dirs=[numpy.get_include()]
)    