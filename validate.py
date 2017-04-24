import numpy as np
np.set_printoptions(threshold=np.nan)

import sgf

import AlphaGo.go as goFast
from AlphaGo.preprocessing.preprocessing import Preprocess as PreprocessFast

import AlphaGo.go_slow as goSlow
from AlphaGo.preprocessing.preprocessing_slow import Preprocess as PreprocessSlow

available = [ "board", "ones", "liberties", "capture_size", "turns_since" ]
test      = []
size      = 5

state_fast = goFast.GameState( size = size )
state_fast.do_move( ( 2, 2 ) )

state_slow = goSlow.GameState( size = size )
state_slow.do_move( ( 2, 2 ) )

correct = 0
countMoves = 0

# test all of them separately
for feature in available:
    
    test = [ feature ]
    print( test )
        
    preproces_slow = PreprocessSlow( test )
    preproces_fast = PreprocessFast( test )
    
    tensor_slow    = preproces_slow.state_to_tensor( state_slow )
    tensor_fast    = preproces_fast.state_to_tensor( state_fast )

    equal = np.array_equal( tensor_slow, tensor_fast )

    print( equal )
    
    if not equal:
        print( tensor_slow )
        print( tensor_fast )
    print( "" )

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

def test_sgf( filepath, features ):

    global correct
    global countMoves

    currmove = 0
 
    preproces_slow = PreprocessSlow( [ features ] )
    preproces_fast = PreprocessFast( [ features ] )

    # new 19*19 board
    state_fast = goFast.GameState()
    state_slow = goSlow.GameState()

    # validate board play is correct with SGF
    with open( filepath, 'r') as file_object:

        collection = sgf.parse( file_object.read() )

    game = collection[0]
    if game.rest is not None:
        for node in game.rest:
            props = node.properties
            if 'W' in props:
                move = _parse_sgf_move(props['W'][0])
                player = goSlow.WHITE
            elif 'B' in props:
                move = _parse_sgf_move(props['B'][0])
                player = goSlow.BLACK
            # update state to n+1

            state_fast.do_move( move )
            state_slow.do_move( move, player )


            tensor_slow    = preproces_slow.state_to_tensor( state_slow )
            tensor_fast    = preproces_fast.state_to_tensor( state_fast )

            equal = np.array_equal( tensor_slow, tensor_fast )
            if not equal:
                #print( str( equal ) + " move " + str( move ) + " " + str( currmove ) )
                #print( tensor_slow )
                #print( tensor_fast )
                #state_fast.printer()
                #sprint( ( tensor_fast - tensor_slow ) )
                #break
                n = 1
            else:
                correct = correct + 1

            countMoves = countMoves + 1
            currmove   = currmove   + 1

available = [ "board", "ones", "liberties", "capture_size", "turns_since" ]
#available = [ "turns_since" ]
sgfFiles  = [ "tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf",
              "tests/test_data/sgf/20160313-AlphaGo-vs-Lee-Sedol.sgf",
              "tests/test_data/sgf/AlphaGo-vs-Lee-Sedol-20160310.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf" ]
#sgfFiles  = [ "tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf" ]

for feature in available:
    for sgfFile in sgfFiles:

        correct = 0
        countMoves = 0
        test_sgf( sgfFile, feature )
    print( feature + " - moves: " + str( countMoves ) + " correct: " + str( correct ) )
    print( "" )