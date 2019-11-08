from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(
        "sporf_transform",
        sources=["_project_sporf.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]


setup(name="project_sporf", ext_modules=cythonize(extensions))
