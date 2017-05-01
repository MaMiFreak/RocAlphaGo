import time
import ast
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from AlphaGo.go cimport GameState, Group, BLACK, EMPTY

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
    cdef int get_board( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
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
    cdef int get_turns_since( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A feature encoding the age of the stone at each location up to 'maximum'

           Note:
           - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
           - EMPTY locations are all-zero features
        """

        cdef short location
        cdef int   age      = offSet + 7
        cdef int   i        = len( state.history ) - 1
        cdef dict  agesSet  = {}

        # set all stones to max age
        for location in state.history:
            if state.board[ location ] > EMPTY:
                tensor[ age, location ] = 1

        age = 0

        # loop over history backwards
        while age < 7 and i >= 0:

            location = state.history[ i ]

            # if age has not been set yet
            if not location in agesSet and state.board[ location ] > EMPTY:

                tensor[ offSet + age, location ] = 1
                tensor[ offSet + 7,   location ] = 0
                agesSet[ location ]              = location

            i   -= 1
            age += 1

        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_liberties( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
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
    cdef int get_capture_size( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
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

            location = group.location_stones.values()[0]

            # check if group has one liberty and is owned by opponent
            if state.board[ location ] == opponent and len( group.location_liberty ) == 1:

                # calculate group size ( including other groups captured )
                location  = group.location_liberty.values()[0]
                groupSize = state.get_capture_size( location, 7 )

                if groupSize > 0 or len( group.location_liberty ) > 1:

                    tensor[ offSet + groupSize, location ] = 1

                tensor[ offSet, location ] = 0
            
        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_self_atari_size( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A feature encoding the size of the own-stone group that is put into atari by
           playing at a location

        """

        cdef Group group
        cdef short location
        cdef int   group_liberty

        # loop over all groups on board
        for location in state.legalMoves:

            group = groups_after[ location ]
            group_liberty = len( group.location_liberty )
            if group_liberty == 1:

                group_liberty = len( group.location_stones ) - 1
                if group_liberty > 7:
                    group_liberty = 7

                tensor[ offSet + group_liberty, location ] = 1
                    
        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_liberties_after( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A feature encoding what the number of liberties *would be* of the group connected to
           the stone *if* played at a location

           Note:
           - there is no zero-liberties plane; the 0th plane indicates groups in atari
           - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
           - illegal move locations are all-zero features
        """

        cdef short location
        cdef Group group
        cdef int   liberty

        # loop over all legal moves
        for location in state.legalMoves:

            group = groups_after[ location ]

            liberty = len( group.location_liberty ) - 1

            if liberty > 7:
                liberty = 7

            if liberty >= 0:

                tensor[ offSet + liberty, location ] = 1

        return offSet + 8

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_ladder_capture( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A feature wrapping GameState.is_ladder_capture().
           check if an opponent group can be captured in a ladder
        """

        cdef Group group
        cdef short location
        cdef char  opponent = state.player_opponent

        # loop over all groups on board
        for group in state.groupsList:

            location = group.location_stones.values()[0]

            # check if group has one liberty and is owned by opponent
            if state.board[ location ] == opponent and len( group.location_liberty ) == 2:

                for location in group.location_liberty:

                    if tensor[ offSet, location ] == 0:
                        tensor[ offSet, location ] = state.is_ladder_capture( group, location, 80 )

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_ladder_escape( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A feature wrapping GameState.is_ladder_escape().
           check if player_current group can escape ladder
        """

        cdef Group group
        cdef short location
        cdef char  current = state.player_current

        # loop over all groups on board
        for group in state.groupsList:

            location = group.location_stones.values()[0]

            # check if group has one liberty and is owned by opponent
            if state.board[ location ] == current and len( group.location_liberty ) == 1:

                location = group.location_liberty.values()[0]

                if tensor[ offSet, location ] == 0:
                    tensor[ offSet, location ] = state.is_ladder_escape( group, location, 80 )

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_sensibleness( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
        """

        cdef short location
        cdef list  sensible_moves = state.get_sensible_moves()

        # loop over all sensible moves and set to 1
        for location in sensible_moves:

            tensor[ offSet, location ] = 1

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_legal( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
           not used??
        """

        cdef short location

        # loop over all legal moves and set to one
        for location in state.legalMoves:

            tensor[ offSet, location ] = 1

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_response( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_save_atari( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_neighbor( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        # get last move location
        # check for pass

        return offSet + 2

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_nakade( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        return offSet + self.pattern_nakade_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_response_12d( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        # get last move location
        # check for pass

        return offSet + self.pattern_response_12d_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int get_non_response_3x3( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Fast rollout feature
        """

        return offSet + self.pattern_non_response_3x3_size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int zeros( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Plane filled with zeros
        """
        
        #########################################################
        # strange things happen if a function does no do anything
        # do not remove next line without extensive testing!!!!!!
        tensor[ offSet, 0 ] = 0

        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int ones( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Plane filled with ones
        """

        cdef short location

        for location in range( 0, self.board_locations ):

            tensor[ offSet, location ] = 1
        return offSet + 1

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef int colour( self, GameState state, tensor_type[ :, ::1 ] tensor, list groups_after, int offSet ):
        """
           Value net feature, plane with ones if active_player is black else zeros
        """

        cdef short location

        # if player_current is white
        if state.player_current == BLACK:

                for location in range( 0, self.board_locations ):

                    tensor[ offSet, location ] = 1

        return offSet + 1

    ############################################################################
    #   init function                                                          #
    #                                                                          #
    ############################################################################

    def __init__( self, list feature_list, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False ):
        """
        """

        cdef int i
        cdef preprocess_method processor
        self.processors = <preprocess_method  *>malloc( len( feature_list ) * sizeof( preprocess_method  ) )

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

        self.feature_list = feature_list
        self.output_dim = 0

        for i in range( len( feature_list ) ):
            feat = feature_list[ i ].lower()
            if feat == "board":
                processor            = self.get_board
                self.output_dim     += 3

            elif feat == "ones":
                processor            = self.ones
                self.output_dim     += 1

            elif feat == "turns_since":
                processor            = self.get_turns_since
                self.output_dim     += 8

            elif feat == "liberties":
                processor            = self.get_liberties
                self.output_dim     += 8

            elif feat == "capture_size":
                processor            = self.get_capture_size
                self.output_dim     += 8

            elif feat == "self_atari_size":
                processor            = self.get_self_atari_size
                self.output_dim     += 8

            elif feat == "liberties_after":
                processor            = self.get_liberties_after
                self.output_dim     += 8

            elif feat == "ladder_capture":
                processor            = self.get_ladder_capture
                self.output_dim     += 1

            elif feat == "ladder_escape":
                processor            = self.get_ladder_escape
                self.output_dim     += 1

            elif feat == "sensibleness":
                processor            = self.get_sensibleness
                self.output_dim     += 1

            elif feat == "zeros":
                processor            = self.zeros
                self.output_dim     += 1

            elif feat == "legal":
                processor            = self.get_legal
                self.output_dim     += 1

            elif feat == "response":
                processor            = self.get_response
                self.output_dim     += 1

            elif feat == "save_atari":
                processor            = self.get_save_atari
                self.output_dim     += 1

            elif feat == "neighbor":
                processor            = self.get_neighbor
                self.output_dim     += 2

            elif feat == "nakade":
                processor            = self.get_nakade
                self.output_dim     += self.pattern_nakade_size

            elif feat == "response_12d":
                processor            = self.get_response_12d
                self.output_dim     += self.pattern_response_12d_size

            elif feat == "non_response_3x3":
                processor            = self.get_non_response_3x3
                self.output_dim     += self.pattern_non_response_3x3_size

            elif feat == "color":
                processor            = self.colour
                self.output_dim     += 1
            else:
                raise ValueError( "uknown feature: %s" % feat )

            self.processors[ i ] = processor

    # deallocate all arrays
    # arrays created with malloc have to be freed when this instance is destroyed
    def __dealloc__(self):

        if self.processors is not NULL:
            free( self.processors )

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef np.ndarray[ tensor_type, ndim=4 ] generate_tensor( self, GameState state ):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        cdef int i
        cdef preprocess_method proc

        cdef char size       = state.size
        self.board_locations = state.size * state.size

        # create complete array now instead of concatenate later
        cdef np.ndarray[ tensor_type, ndim=2 ] np_tensor = np.zeros( ( self.output_dim, self.board_locations ), dtype=np.int8 )
        cdef tensor_type[ :, ::1 ] tensor                = np_tensor

        cdef int offSet = 0

        cdef list groups_after = state.get_groups_after()
        # TODO create array with all nextmoves information

        for i in range( len( self.feature_list ) ):

            proc   = self.processors[ i ]
            offSet = proc( self, state, tensor, groups_after, offSet )


        # create a singleton 'batch' dimension
        return np_tensor.reshape( ( 1, self.output_dim, size, size ) )


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

        import time
        t = time.time()
 
        cdef int i

        for i in range( amount ):
            self.generate_tensor( state )

        print "proc " + str( time.time() - t )

    def timed_test( self, GameState state, int amount ):

        cdef int i

        for i in range( amount ):

            self.generate_tensor( state )


    def test_game_speed( self, GameState state, list moves ):

        cdef short location

        for location in moves:

            state.add_move( location )
            self.generate_tensor( state )