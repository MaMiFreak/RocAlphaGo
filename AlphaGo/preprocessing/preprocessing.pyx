import time
import ast
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from AlphaGo.go cimport GameState, Group

cdef class Preprocess:

    ############################################################################
    #   all variables are declared in the .pxd file                            #
    #                                                                          #
    ############################################################################

    """ -> variables, declared in preprocessing.pxd

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

        -> variables, declared in preprocessing.pxd
    """

    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_board( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """
           A feature encoding WHITE BLACK and EMPTY on separate planes.
           plane 0 always refers to the current player stones
           plane 1 to the opponent stones
           plane 2 to empty locations
        """

        cdef short location

        # loop over all locations on board
        for location in range( 0, self.board_locations ):

            tensor[ offSet + state.get_board_feature( location ), location ] = 1

        return offSet + 3

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_turns_since( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """
           A feature encoding the age of the stone at each location up to 'maximum'

           Note:
           - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
           - EMPTY locations are all-zero features
        """

        cdef short location
        cdef int   age = 0
        cdef int   i   = len( state.history ) - 1

        # loop over history backwards
        while i >= 0:

            location = state.history[ i ]

            # set age
            tensor[ offSet + age, location ] = 1

            i -= 1

            if age < 7:
                age += 1 

        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_liberties( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """
           A feature encoding the number of liberties of the group connected to the stone at
           each location

           Note:
           - there is no zero-liberties plane; the 0th plane indicates groups in atari
           - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
           - EMPTY locations are all-zero features
        """

        cdef Group group
        cdef short location
        cdef int   groupLiberty

        # loop over all groups on board
        for group in state.groupsList:

            # calculate liberty
            groupLiberty = len( group.location_liberty ) - 1

            # check max
            if groupLiberty > 7:
                groupLiberty = 7

            # loop over all group stones and set liberty count
            for location in group.location_stones:

                tensor[ offSet + groupLiberty, location ] = 1
            
        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_capture_size( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """
           A feature encoding the number of opponent stones that would be captured by
           playing at each location, up to 'maximum'

           Note:
           - we currently *do* treat the 0th plane as "capturing zero stones"
           - the [maximum-1] plane is used for any capturable group of size
             greater than or equal to maximum-1
           - the 0th plane is used for legal moves that would not result in capture
           - illegal move locations are all-zero features
        """

        cdef Group group
        cdef short location
        cdef int   groupSize
        cdef char  opponent = state.player_opponent

        # loop over all legal moves and set to zero
        for location in state.legalMoves:

            tensor[ offSet, location ] = 1

        # loop over all groups on board
        for group in state.groupsList:

            # check if group has one liberty
            if len( group.location_liberty ) == 1 and state.board[group.location_liberty.values()[0]] == opponent:

                # calculate group size
                groupSize = len( group.location_stones ) - 1

                # check max
                if groupSize > 7:
                    groupSize = 7

                location = group.location_liberty.values()[0]
                tensor[ offSet + groupSize, location ] = 1
                tensor[ offSet, location ] = 0
            
        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_self_atari_size( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A feature encoding the size of the own-stone group that is put into atari by
        playing at a location

        """

        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_liberties_after( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A feature encoding what the number of liberties *would be* of the group connected to
        the stone *if* played at a location

        Note:
        - there is no zero-liberties plane; the 0th plane indicates groups in atari
        - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
        - illegal move locations are all-zero features
        """

        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_ladder_capture( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A feature wrapping GameState.is_ladder_capture().
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_ladder_escape( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A feature wrapping GameState.is_ladder_escape().
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_sensibleness( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_legal( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_response( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_save_atari( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_neighbor( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        # get last move location
        # check for pass

        return offSet + 2

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_nakade( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        return offSet + self.pattern_nakade_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_response_12d( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        # get last move location
        # check for pass

        return offSet + self.pattern_response_12d_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_non_response_3x3( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        return offSet + self.pattern_non_response_3x3_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int zeros( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """
        
        # do nothing, numpy array has to be initialized with zero

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int ones( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        cdef short location

        for location in range( 0, self.board_locations ):

            tensor[ offSet, location ] = 1
        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int colour( self, GameState state, np.ndarray[double, ndim=2] tensor, int offSet ):
        """Fast rollout feature
        """

        return offSet + 1

    ############################################################################
    #   init function                                                          #
    #                                                                          #
    ############################################################################

    def __init__( self, feature_list, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False ):
        """
        """

        # load nakade patterns
        self.pattern_nakade = {}
        self.pattern_nakade_size = 0
        if dict_nakade is not None:
            with open(dict_nakade, 'r') as f:
                s = f.read()
                self.pattern_nakade = ast.literal_eval(s)
                self.pattern_nakade_size = max(self.pattern_nakade.values()) + 1
        
        # load 12d response patterns
        self.pattern_response_12d = {}
        self.pattern_response_12d_size = 0
        if dict_12d is not None:
            with open(dict_12d, 'r') as f:
                s = f.read()
                self.pattern_response_12d = ast.literal_eval(s)
                self.pattern_response_12d_size = max(self.pattern_response_12d.values()) + 1

        # load 3x3 non response patterns
        self.pattern_non_response_3x3 = {}
        self.pattern_non_response_3x3_size = 0
        if dict_3x3 is not None:
            with open(dict_3x3, 'r') as f:
                s = f.read()
                self.pattern_non_response_3x3 = ast.literal_eval(s)
                self.pattern_non_response_3x3_size = max(self.pattern_non_response_3x3.values()) + 1
        
        if verbose:
            print("loaded " + str(self.pattern_nakade_size) + " nakade patterns")
            print("loaded " + str(self.pattern_response_12d_size) + " 12d patterns")
            print("loaded " + str(self.pattern_non_response_3x3_size) + " 3x3 patterns")

        # TODO this is slow, find correct cdef types 
        # named features and their sizes are defined here
        FEATURES = {
            "board": {
                "size": 3,
                "function": self.get_board
            },
            "ones": {
                "size": 1,
                "function": self.ones
            },
            "turns_since": {
                "size": 8,
                "function": self.get_turns_since
            },
            "liberties": {
                "size": 8,
                "function": self.get_liberties
            },
            "capture_size": {
                "size": 8,
                "function": self.get_capture_size
            },
            "self_atari_size": {
                "size": 8,
                "function": self.get_self_atari_size
            },
            "liberties_after": {
                "size": 8,
                "function": self.get_liberties_after
            },
            "ladder_capture": {
                "size": 1,
                "function": self.get_ladder_capture
            },
            "ladder_escape": {
                "size": 1,
                "function": self.get_ladder_escape
            },
            "sensibleness": {
                "size": 1,
                "function": self.get_sensibleness
            },
            "zeros": {
                "size": 1,
                "function": self.zeros
            },
            "legal": {
                "size": 1,
                "function": self.get_legal
            },
            "response": {
                "size": 1,
                "function": self.get_response
            },
            "save_atari": {
                "size": 1,
                "function": self.get_save_atari
            },
            "neighbor": {
                "size": 2,
                "function": self.get_neighbor
            },
            "nakade": {
                "size": self.pattern_nakade_size,
                "function": self.get_nakade
            },
            "response_12d": {
                "size": self.pattern_response_12d_size,
                "function": self.get_response_12d
            },
            "non_response_3x3": {
                "size": self.pattern_non_response_3x3_size,
                "function": self.get_non_response_3x3
            },
            "color": {
                "size": 1,
                "function": self.colour
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

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################

    cdef np.ndarray[ double, ndim=3 ] generate_tensor( self, GameState state ):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        self.board_locations = state.size * state.size
        cdef char size = state.size

        # create complete array now instead of concatenate later
        cdef np.ndarray[ double, ndim = 2 ] tensor = np.zeros( ( self.output_dim, self.board_locations ) )
        cdef int offSet = 0

        for proc in self.processors:

            offSet = proc( self, state, tensor, offSet )


        # create a singleton 'batch' dimension
        return tensor.reshape( ( 1, self.output_dim, size, size ) )


    ############################################################################
    #   public def function (Python)                                           #
    #                                                                          #
    ############################################################################


    # this function should be used from Python environment, 
    # use generate_tensor from C environment for speed
    def state_to_tensor( self, GameState state ):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        return self.generate_tensor( state )


    def test( self, GameState state, int amount ):
        cdef char size = state.size
        self.board_locations = state.size * state.size

        # create complete array now instead of concatenate later
        cdef np.ndarray[ double, ndim = 2 ] tensor = np.zeros( ( self.output_dim, self.board_locations ) )
        cdef int offSet = 0

        import time
        t = time.time()
 
        cdef int i

        for i in range( amount ):
            for proc in self.processors:

                offSet = proc( self, state, tensor, 0 )

        print "proc " + str( time.time() - t )

    def timed_test( self, GameState state, int amount ):

        cdef int i

        for i in range( amount ):

            self.generate_tensor( state )