import pyximport; pyximport.install()
import time
import numpy
import Cython
# build_ext --inplace

import AlphaGo.go as go
from AlphaGo.preprocessing.preprocessing import Preprocess

print numpy.__version__
print Cython.__version__

state = go.GameState()
state.do_move( (10, 10) )

prep = Preprocess( [ "board" ] )
for _ in range(10):
    start = time.time()

    for _ in range(100000):
        prep.state_to_tensor(state)

    print( time.time() - start )
