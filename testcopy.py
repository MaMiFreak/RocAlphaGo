import numpy as np
np.set_printoptions(threshold=np.nan)

import time
import sys

import sgf

import AlphaGo.go as goFast
from AlphaGo.preprocessing.preprocessing import Preprocess as PreprocessFast

available = [ "board", "ones", "turns_since", "liberties", "capture_size",
              "self_atari_size", "liberties_after", "ladder_capture",
              "ladder_escape", "sensibleness", "zeros", "color", "legal" ]
# available = [ "board" ]

sgfFiles  = [ "tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf",
              "tests/test_data/sgf/20160313-AlphaGo-vs-Lee-Sedol.sgf",
              "tests/test_data/sgf/AlphaGo-vs-Lee-Sedol-20160310.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf",
              "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf" ]
#sgfFiles  = [ "tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160315.sgf" ]

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
    
def get_moves( filepath ):

    moves = []

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
            
            moves.append( move )
            
    return moves


count = 0
correct = 0

# 
#for feature in available:
start = time.time()

for sgfFile in sgfFiles:

    preproces = PreprocessFast( available )

    # new 19*19 board
    state_root = goFast.GameState()
    moves = get_moves( sgfFile )

    for i in range( len( moves ) ):

        state_root.do_move( moves[ i ] )

        tensor_root = preproces.state_to_tensor( state_root )

        # print( tensor_root.nbytes )
        # print( tensor_root.shape )
        # print sys.getrefcount( tensor_root )

        state_copy  = state_root.copy()
        
        # check if copy is equal
        tensor_validate  = preproces.state_to_tensor( state_copy )
        count += 1
        equal = np.array_equal( tensor_root, tensor_validate )
        if not equal:
            print( "tensor not equal " + str( i ) + " - " + str( a ) )
        else:
            correct += 1
        
        for a in range( i + 1, len( moves ) ):

            
            state_copy.do_move( moves[ a ] )
            count += 1

            # validate root state has not changed
            tensor_validate  = preproces.state_to_tensor( state_root )

            # check if original is unchanged
            equal = np.array_equal( tensor_root, tensor_validate )
            
            del tensor_validate

            if not equal:
                print( "tensor not equal " + str( i ) + " - " + str( a ) )
            else:
                correct += 1
                
        del state_copy
                    
print( "tests: " + str( count ) + " correct: " + str( correct ) )
print( "time: " + str( time.time() - start ) )




        