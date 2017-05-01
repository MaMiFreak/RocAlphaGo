from timeit import timeit

import sgf

import AlphaGo.go as go_cython
from AlphaGo.preprocessing.preprocessing import Preprocess as Preprocess_cython

import AlphaGo.go_python as go_python
from AlphaGo.preprocessing.preprocessing_python import Preprocess as Preprocess_python

amount    = 100
bestof    = 10
bestofsgf = 40
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

# time a funcion call #amount times and return cythones call
def time_repeat( call, amount ):
    timed = 1000000
    for _ in range( bestof ):
        timed = min( timeit( call, number = amount, ), timed )
    return timed
        
state_cython_root = go_cython.GameState()
        
state_cython = state_cython_root.copy()
state_cython.do_move( (10, 10) )

state_python = go_python.GameState()
state_python.do_move( (10, 10) )

def timefeature( features ):

    preproces_python = Preprocess_python( features )
    preproces_cython = Preprocess_cython( features )
    
    # pure python implementation
    def python():
        preproces_python.state_to_tensor( state_python )
        
    # cython implementation but with #amount python function calls to start tensor creation
    def cython():
        preproces_cython.state_to_tensor( state_cython )
        
    # cython implementation with 1 python function call to start test in pure C environment
    def pure():
        preproces_cython.timed_test( state_cython, amount )
    
    time_python = time_repeat( python, amount )
    print( "Python : " + str( time_python ) )
    
    time_cython = time_repeat( cython, amount )
    print( "Cython-: " + str( time_cython ) + " speedup: " + str( round( time_python / time_cython *10 ) / 10 ) + "X" )
    
    time_pure = time_repeat( pure, 1 )
    print( "Cython+: " + str( time_pure ) + " speedup: " + str( round( time_python / time_pure *10 ) / 10 ) + "X" )
     
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
        return go_python.PASS_MOVE
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

def cython_game():
    for moves in test:
        state_cython.do_move( moves )

def super_cython_game():
    state_cython.test_game_speed( test )

def python_game():
    for mover in test:
        state_python.do_move( mover )

# time a funcion call #amount times and return cythones call
def time_repeat_cython_sgf( call ):
    global state_cython
    timed = 1000000
    for _ in range( bestofsgf ):
        state_cython = state_cython_root.copy()
        timed = min( timeit( call, number = 1, ), timed )
    return timed
                        
for singleFile in sgfFiles:
    
    test = []
    convert_sgf( singleFile )
    
    state_cython = state_cython_root.copy()
    state_python = go_python.GameState()   
                
    print( "do moves: " + str( len( test ) ) )
    
    time_python = timeit( python_game, number = 1, )
    print( "Python : " + str( time_python ) )

    time_cython = time_repeat_cython_sgf( cython_game )
    print( "Cython-: " + str( time_cython ) + " speedup: " + str( round( time_python / time_cython *10 ) / 10 ) + "X" )
    
    state_cython = state_cython_root.copy()
    test = state_cython.convert_moves( test )
    time_cython = time_repeat_cython_sgf( super_cython_game )
    print( "Cython+: " + str( time_cython ) + " speedup: " + str( round( time_python / time_cython *10 ) / 10 ) + "X" )
    print( "" )

print( available )
    
for singleFile in sgfFiles:    
    # play game and convert
    
    preproces_python = Preprocess_python( available )
    preproces_cython = Preprocess_cython( available )
    
    state_cython = state_cython_root.copy()
    state_python = go_python.GameState()   
        
    test = []
    convert_sgf( singleFile )
    
    def cython_game_convert():
        for moves in test:
            state_cython.do_move( moves )
            preproces_cython.state_to_tensor( state_cython )
            
    def super_cython_game_convert():
        preproces_cython.test_game_speed( state_cython, test )

    def python_game_convert():
        for mover in test:
            state_python.do_move( mover )
            preproces_python.state_to_tensor( state_python )
    
    print( "do moves and convert sgf: " + str( len( test ) ) )
    
    time_python = timeit( python_game_convert, number = 1, )
    print( "Python : " + str( time_python ) )

    time_cython = timeit( cython_game_convert, number = 1, )
    print( "Cython-: " + str( time_cython ) + " speedup: " + str( round( time_python / time_cython *10 ) / 10 ) + "X" )
    
    state_cython = state_cython_root.copy()
    test = state_cython.convert_moves( test )
    time_cython = timeit( super_cython_game_convert, number = 1, )
    print( "Cython+: " + str( time_cython ) + " speedup: " + str( round( time_python / time_cython *10 ) / 10 ) + "X" )

    print( "" )    
    
    