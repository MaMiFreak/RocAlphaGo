import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    
    name = 'RocAlphaGo',
    # list with files to be cythonized
    ext_modules = cythonize(["AlphaGo/go.pyx", "AlphaGo/preprocessing/preprocessing.pyx"]),
    # include numpy
    include_dirs=[numpy.get_include()]
)

# run setup with command
# python setup.py build_ext --inplace

# be aware cython uses a depricaped version of numpy this results in a lot of warnings