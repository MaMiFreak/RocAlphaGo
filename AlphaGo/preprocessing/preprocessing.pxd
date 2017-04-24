import time
import ast
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from AlphaGo.go cimport GameState, Group

cdef class Preprocess:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    # all feature processors
    # TODO find correct type so an array can be used
    cdef list  processors

    # list with all features used currently
    # TODO find correct type so an array can be used
    cdef list  feature_list

    # output tensor size
    cdef int   output_dim

    # board size
    cdef short board_locations

    # pattern dictionaries
    cdef dict  pattern_nakade
    cdef dict  pattern_response_12d
    cdef dict  pattern_non_response_3x3

    # pattern dictionary sizes
    cdef int   pattern_nakade_size
    cdef int   pattern_response_12d_size
    cdef int   pattern_non_response_3x3_size

    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################

    cdef int get_board( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_turns_since( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_liberties( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_capture_size( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_self_atari_size( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_liberties_after( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_ladder_capture( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_ladder_escape( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_sensibleness( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_legal( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_response( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_save_atari( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_neighbor( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_nakade( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_response_12d( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int get_non_response_3x3( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int zeros( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int ones( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )
    cdef int colour( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet )

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################

    cdef np.ndarray[ double, ndim=3 ] generate_tensor( self, GameState state )
