import numpy as np
from random import shuffle
import os
np.set_printoptions(threshold=np.nan)

import sgf

import AlphaGo.go as goFast
from AlphaGo.preprocessing.preprocessing import Preprocess as PreprocessFast

import AlphaGo.go_slow as goSlow
from AlphaGo.preprocessing.preprocessing_python import Preprocess as PreprocessSlow

available = [ "board", "ones", "turns_since", "liberties", "capture_size",
              "self_atari_size", "liberties_after", "ladder_capture",
              "ladder_escape", "sensibleness", "zeros", "color", "legal" ]
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

    currmove = 1
 
    preproces_slow = PreprocessSlow( features )
    preproces_fast = PreprocessFast( features )

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
            state_slow.do_move( move )


            tensor_slow    = preproces_slow.state_to_tensor( state_slow )
            tensor_fast    = preproces_fast.state_to_tensor( state_fast )

            equal = np.array_equal( tensor_slow, tensor_fast )
            if not equal:
                if True:
                    """
                    print( tensor_slow )
                    print("")
                    print( tensor_fast )
                    print("")
                    state_fast.printer()
                    print("")
                    print( ( tensor_fast - tensor_slow ) )
                    print("")
                    print( str( equal ) + " move " + str( move ) + " " + str( currmove ) )
                    print( state_fast.get_player_active_colour() )"""
                    #state_fast.printer()
                    #print( ( tensor_fast - tensor_slow ) )
                    print( filepath )
                    print( str( equal ) + " move " + str( move ) + " " + str( currmove ) )
                    break
                    
            else:
                correct = correct + 1

            countMoves = countMoves + 1
            currmove   = currmove   + 1

#available = [ "zeros", "legal" ] 
sgfFiles  = [ "tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf",
              "tests/test_data/sgf/20160313-AlphaGo-vs-Lee-Sedol.sgf",
              "tests/test_data/sgf/AlphaGo-vs-Lee-Sedol-20160310.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf",
              "/media/boss/2C260F93260F5CE8/Godatabasedata/nohan/1200.sgf" ]
# sgfFiles  = [ "/media/boss/2C260F93260F5CE8/Godatabasedata/nohan/1200.sgf" ]

def _is_sgf(fname):
    return fname.strip()[-4:] == ".sgf"
    
def _list_sgfs(path):
    """helper function to get all SGF files in a directory (does not recurse)
    """
    
    file_names = []
    
    files     = os.listdir(path)
    files_sgf = (os.path.join(path, f) for f in files if _is_sgf(f))
    
    for name in files_sgf:
        file_names.append( name )
    
    shuffle( file_names )
    
    return file_names

folder_sgf = "/media/boss/2C260F93260F5CE8/Godatabasedata/nohan"
#folder_sgf = "/media/boss/2C260F93260F5CE8/Godatabasedata/pro/alphago"
#sgfFiles   = _list_sgfs( folder_sgf )

correct = 0
countMoves = 0

for sgfFile in sgfFiles:

    test_sgf( sgfFile, available )
    
    print( "moves: " + str( countMoves ) + " correct: " + str( correct ) )
print( available )
print( "" )

for feature in available:

    correct = 0
    countMoves = 0
    
    for sgfFile in sgfFiles:

        n = 1
        test_sgf( sgfFile, [feature] )
    print( "moves: " + str( countMoves ) + " correct: " + str( correct ) + " - " + feature )
    print( "" )
    
    