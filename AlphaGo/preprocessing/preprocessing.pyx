import time
import ast
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from AlphaGo.go cimport GameState

cdef class Preprocess:

    # all feature processors
    cdef object processors

    # list with all features used currently
    cdef object feature_list

    # output tensor size
    cdef int    output_dim

    # pattern dictionaries
    cdef dict   pattern_nakade
    cdef dict   pattern_response_12d
    cdef dict   pattern_non_response_3x3

    # pattern dictionary sizes
    cdef int    pattern_nakade_size
    cdef int    pattern_response_12d_size
    cdef int    pattern_non_response_3x3_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get_board(self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
           always refers to the current player and plane 1 to the opponent
        """

        cdef short location

        for location in range( 0, state.size * state.size ):

            tensor[ offSet + state.get_board_feature( location ), location ] = 1

        return offSet + 3


    def __init__(self, feature_list, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False):
        """
        """

        # named features and their sizes are defined here
        FEATURES = {
            "board": {
                "size": 3,
                "function": self.get_board
            }
        }

        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)

        for i in range( len( feature_list ) ):
            feat = feature_list[ i ].lower()
            if feat in FEATURES:
                self.processors[ i ] = FEATURES[ feat ][ "function" ]
                self.output_dim     += FEATURES[ feat ][ "size"     ]
            else:
                raise ValueError( "uknown feature: %s" % feat )


    def state_to_tensor( self, GameState state ):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        cdef char size = state.size

        # create complete array now instead of concatenate later
        cdef np.ndarray[ double, ndim = 2 ] tensor = np.zeros( ( self.output_dim, size * size ) )
        cdef int offSet = 0

        for proc in self.processors:

            offSet = proc( self, state, tensor, offSet )


        # create a singleton 'batch' dimension
        return tensor.reshape( ( 1, self.output_dim, size, size ) )
