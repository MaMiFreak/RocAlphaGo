from timeit import timeit

import sgf

import AlphaGo.go as goFast
from AlphaGo.preprocessing.preprocessing import Preprocess as PreprocessFast

import AlphaGo.go_slow as goSlow
from AlphaGo.preprocessing.preprocessing_slow import Preprocess as PreprocessSlow

amount    = 10
bestof    = 10
available = [ "board", "ones", "turns_since", "liberties", "capture_size",
              "self_atari_size", "liberties_after", "ladder_capture",
              "ladder_escape", "sensibleness", "zeros", "color" ]
#available = [ "board", "turns_since", "liberties", "capture_size", "self_atari_size", "liberties_after" ]

sgfFiles  = [ "tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf",
              "tests/test_data/sgf/20160313-AlphaGo-vs-Lee-Sedol.sgf",
              "tests/test_data/sgf/AlphaGo-vs-Lee-Sedol-20160310.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf" ]
#sgfFiles   = [ "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf" ]
test       = []

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
     
# test all of them
print( available )    
timefeature( available )
print( "" )    
    
# test all of them separately
for feature in available:
    
    test = [ feature ]
    print( test )
    
    timefeature( test )
    
    print( "" )
    
# test a game    

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _parse_sgf_move(node_value):
    """Given a well-formed move string, return either PASS_MOVE or the (x, y) position
    """
    if node_value == '' or node_value == 'tt':
        return goSlow.PASS_MOVE
    else:
        # GameState expects (x, y) where x is column and y is row
        col = LETTERS.index(node_value[0].upper())
        row = LETTERS.index(node_value[1].upper())
        return (col, row)
    
test = []

def convert_sgf( filepath ):
    # validate board play is correct with SGF
    with open( filepath, 'r') as file_object:

        collection = sgf.parse( file_object.read() )

    game = collection[0]
    if game.rest is not None:
        for node in game.rest:
            props = node.properties
            if 'W' in props:
                move = _parse_sgf_move(props['W'][0])
            elif 'B' in props:
                move = _parse_sgf_move(props['B'][0])
            # update state to n+1

            test.append( move )
    
                        
for singleFile in sgfFiles:
    
    test = []
    convert_sgf( singleFile )
    
    state_fast = goFast.GameState()
    state_slow = goSlow.GameState()   
            
    def fast_game():
        for moves in test:
            state_fast.do_move( moves )
            
    def super_fast_game():
        state_fast.test_game_speed( test )

    def slow_game():
        for mover in test:
            state_slow.do_move( mover )
            
    
    print( "do moves: " + str( len( test ) ) )
    
    time_slow = timeit( slow_game, number = 1, )
    print( "Python : " + str( time_slow ) )

    time_fast = timeit( fast_game, number = 1, )
    print( "Cython-: " + str( time_fast ) + " speedup: " + str( round( time_slow / time_fast *10 ) / 10 ) + "X" )
    
    state_fast = goFast.GameState()
    test = state_fast.convert_moves( test )
    time_fast = timeit( super_fast_game, number = 1, )
    print( "Cython+: " + str( time_fast ) + " speedup: " + str( round( time_slow / time_fast *10 ) / 10 ) + "X" )
    print( "" )

print( available )
    
for singleFile in sgfFiles:    
    # play game and convert
    
    preproces_slow = PreprocessSlow( available )
    preproces_fast = PreprocessFast( available )
    
    state_fast = goFast.GameState()
    state_slow = goSlow.GameState()   
        
    test = []
    convert_sgf( singleFile )
    
    def fast_game_convert():
        for moves in test:
            state_fast.do_move( moves )
            preproces_fast.state_to_tensor( state_fast )
            
    def super_fast_game_convert():
        preproces_fast.test_game_speed( state_fast, test )

    def slow_game_convert():
        for mover in test:
            state_slow.do_move( mover )
            preproces_slow.state_to_tensor( state_slow )
    
    print( "do moves and convert sgf: " + str( len( test ) ) )
    
    time_slow = timeit( slow_game_convert, number = 1, )
    print( "Python : " + str( time_slow ) )

    time_fast = timeit( fast_game_convert, number = 1, )
    print( "Cython-: " + str( time_fast ) + " speedup: " + str( round( time_slow / time_fast *10 ) / 10 ) + "X" )
    
    state_fast = goFast.GameState()
    test = state_fast.convert_moves( test )
    time_fast = timeit( super_fast_game_convert, number = 1, )
    print( "Cython+: " + str( time_fast ) + " speedup: " + str( round( time_slow / time_fast *10 ) / 10 ) + "X" )

    print( "" )    
    
    