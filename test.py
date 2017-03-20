#import pyximport; pyximport.install()
import time
import numpy
import Cython
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'RocAlphaGo',
    ext_modules = cythonize(["AlphaGo/go.pyx", "AlphaGo/preprocessing/preprocessing.pyx"]),
    include_dirs=[numpy.get_include()]
)

# python test.py build_ext --inplace

import AlphaGo.go as go
from AlphaGo.preprocessing.preprocessing import Preprocess

print numpy.__version__
print Cython.__version__

"""
test = Preprocess( [ "board", "color" ] )

state = go.GameState()
#state.printer()
state.do_move( (10, 10) )
#state.printer()
state.test(0)
print()
state.test(20)
print()
state.test(360)
# state.get_next_state( (11, 11) ).printer()

test.state_to_tensor(state)
#test.evalc()
"""

state = go.GameState()
#state.printer()
state.do_move( (10, 10) )

prep = Preprocess( [ "board" ] )

for _ in range(10):

    start = time.time()

    for _ in range(100000):

        prep.state_to_tensor(state)

    print( time.time() - start )
print()
for _ in range(10):
    state.test_speed()