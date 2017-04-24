from timeit import timeit

import AlphaGo.go as goFast
from AlphaGo.preprocessing.preprocessing import Preprocess as PreprocessFast

import AlphaGo.go_slow as goSlow
from AlphaGo.preprocessing.preprocessing_slow import Preprocess as PreprocessSlow

amount    = 200
bestof    = 30
available = [ "board", "ones", "liberties", "capture_size", "turns_since" ]
test      = []

# time a funcion call #amount times and return fastes call
def time_repeat( call, amount ):
    timed = 1000000
    for _ in range( bestof ):
        timed = min( timeit( call, number = amount, ), timed )
    return timed
        
state_fast = goFast.GameState()
state_fast.do_move( (10, 10) )

state_slow = goSlow.GameState()
state_slow.do_move( (10, 10) )

def timefeature( features ):

    preproces_slow = PreprocessSlow( features )
    preproces_fast = PreprocessFast( features )
    
    # pure python implementation
    def slow():
        preproces_slow.state_to_tensor( state_slow )
        
    # cython implementation but with #amount python function calls to start tensor creation
    def fast():
        preproces_fast.state_to_tensor( state_fast )
        
    # cython implementation with 1 python function call to start test in pure C environment
    def pure():
        preproces_fast.timed_test( state_fast, amount )
    
    time_slow = time_repeat( slow, amount )
    print( "Python : " + str( time_slow ) )
    
    time_fast = time_repeat( fast, amount )
    print( "Cython-: " + str( time_fast ) + " speedup: " + str( round( time_slow / time_fast *10 ) / 10 ) + "X" )
    
    time_pure = time_repeat( pure, 1 )
    print( "Cython+: " + str( time_pure ) + " speedup: " + str( round( time_slow / time_pure *10 ) / 10 ) + "X" )
    
# test all of them together
for feature in available:
    
    test.append( feature )
    print( test )
    
    timefeature( test )
    
    print( "" )
    
# test all of them separately
for feature in available:
    
    test = [ feature ]
    print( test )
    
    timefeature( test )
    
    print( "" )
    
    