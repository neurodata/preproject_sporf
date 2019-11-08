from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
        Extension("sporf_transform", sources=["_project_sporf.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
        ]


setup(
    name="_project_sporf",
    ext_modules = cythonize(extensions),
)


