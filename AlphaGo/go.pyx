import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython
import time
from AlphaGo.preprocessing.preprocessing import Preprocess

# observe stones > 0
#         border < 0
# be aware you should NOT use != EMPTY as this includes border locations
cdef char PASS   = -1
cdef char BORDER = -1
cdef char EMPTY  = 0
cdef char WHITE  = 1
cdef char BLACK  = 2

# value used to generate pattern hashes
cdef char HASHVALUE = 33

# structure to store ladder group information
cdef class LadderGroup:

    def __init__( self ):

        self.escape_atari     = {}
        self.location_stones  = {}
        self.location_liberty = {}

# structure to store group information
cdef class Group:

    def __init__( self ):

        self.location_stones  = {}
        self.location_liberty = {}

#    def __richcmp__(Group self, Group other, int op):
#
#        if op==2:
#            print( self.location_stones )
#            print( other.location_stones )
#            print( self.location_stones == other.location_stones )
#            return self.location_stones == other.location_stones and self.location_liberty == other.location_liberty
#        else:
#            err_msg = "op {0} isn't implemented yet".format(op)
#            raise NotImplementedError(err_msg)
        

cdef class GameState:

    ############################################################################
    #   all variables are declared in the .pxd file                            #
    #                                                                          #
    ############################################################################

    """ -> variables, declared in go.pxd

    # size of board side
    cdef char size

    # possible ko location
    cdef short ko                

    # char array representing board locations as char ( EMPTY, WHITE, BLACK )
    cdef char  *board

    ##########################################
    # TODO replace lists with something faster

    # list with all groups
    cdef list groupsList

    # list representing board locations as groups
    # a Group contains all group stone locations and group liberty locations
    cdef list boardGroups

    cdef char player_current
    cdef char player_opponent

    cdef list history

    # array, keep track of 3x3 pattern hashes
    cdef long  *hash3x3

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef list   legalMoves
    cdef dict   hash_lookup
    cdef int    current_hash
    cdef set    previous_hashes

        -> variables, declared in go.pxd
    """

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################


    # return location on board or borderlocation
    # board locations = [ 0, size * size )
    # border location = size * size
    # x = columns
    # y = rows
    cdef short calculate_board_location( self, char x, char y ):

        # check if x or y are outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            # return border location
            return self.size * self.size

        # return board location
        return x + ( y * self.size )


    # create array for every board location with all 4 direct neighbour locations
    # neighbor order: 
    #                 left - right - above - below
    # -1     x 
    #       x x
    # +1     x 
    #
    cdef void set_neighbors( self, int size ):

        # create array
        self.neighbor = <short *>malloc( size * size * 4 * sizeof( short ) )
        cdef short location
        cdef char x, y

        # add all direct neighbors to every board location
        for y in range( size ):
            for x in range( size ):
                location = ( x + ( y * size ) ) * 4
                self.neighbor[ location + 0 ] = self.calculate_board_location( x - 1, y     )
                self.neighbor[ location + 1 ] = self.calculate_board_location( x + 1, y     )
                self.neighbor[ location + 2 ] = self.calculate_board_location( x    , y - 1 )
                self.neighbor[ location + 3 ] = self.calculate_board_location( x    , y + 1 )


    # create for every board location array with all 8 surrounding neighbour locations
    # neighbor order: above left - above middle - above right
    #                 left - right
    #                 below left - below middle - below right
    # -1    xxx
    #       x x
    # +1    xxx
    #
    cdef void set_3x3_neighbors(self, int size):

        # create array
        self.neighbor3x3 = <short *>malloc( size * size * 8 * sizeof( short ) )
        cdef short location
        cdef char x, y

        # add all surrounding neighbors to every board location
        for x in range( size ):
            for y in range( size ):
                location = ( x + ( y * size ) ) * 8
                self.neighbor3x3[ location + 0 ] = self.calculate_board_location( x - 1, y - 1 )
                self.neighbor3x3[ location + 1 ] = self.calculate_board_location( x    , y - 1 )
                self.neighbor3x3[ location + 2 ] = self.calculate_board_location( x + 1, y - 1 )
                self.neighbor3x3[ location + 3 ] = self.calculate_board_location( x - 1, y     )
                self.neighbor3x3[ location + 4 ] = self.calculate_board_location( x + 1, y     )
                self.neighbor3x3[ location + 5 ] = self.calculate_board_location( x - 1, y + 1 )
                self.neighbor3x3[ location + 6 ] = self.calculate_board_location( x    , y + 1 )
                self.neighbor3x3[ location + 7 ] = self.calculate_board_location( x + 1, y + 1 )


    # create array for every board location with 12d star neighbour locations
    # neighbor order: top star tip
    #                 above left - above middle - above right
    #                 left star tip - left - right - right star tip
    #                 below left - below middle - below right
    #                 below star tip
    # 
    # -2     x 
    # -1    xxx
    #      xx xx
    # +1    xxx
    # +2     x
    #
    cdef void set_12d_neighbors( self, int size ):

        # create array
        self.neighbor12d = <short *>malloc( size * size * 12 * sizeof( short ) )
        cdef short location
        cdef char x, y

        # add all 12d neighbors to every board location
        for x in range( size ):
            for y in range( size ):
                location = ( x + ( y * size ) ) * 12
                self.neighbor12d[ location +  4 ] = self.calculate_board_location( x    , y - 2 )

                self.neighbor12d[ location +  1 ] = self.calculate_board_location( x - 1, y - 1 )
                self.neighbor12d[ location +  5 ] = self.calculate_board_location( x    , y - 1 )
                self.neighbor12d[ location +  8 ] = self.calculate_board_location( x + 1, y - 1 )

                self.neighbor12d[ location +  0 ] = self.calculate_board_location( x - 2, y     )
                self.neighbor12d[ location +  2 ] = self.calculate_board_location( x - 1, y     )
                self.neighbor12d[ location +  9 ] = self.calculate_board_location( x + 1, y     )
                self.neighbor12d[ location + 11 ] = self.calculate_board_location( x + 2, y     )

                self.neighbor12d[ location +  3 ] = self.calculate_board_location( x - 1, y + 1 )
                self.neighbor12d[ location +  6 ] = self.calculate_board_location( x    , y + 1 )
                self.neighbor12d[ location + 10 ] = self.calculate_board_location( x + 1, y + 1 )

                self.neighbor12d[ location +  7 ] = self.calculate_board_location( x    , y + 2 )


    # initialize empty board ( root state )
    cdef initialize_new( self, char size ):

        cdef short i
        self.size = size
        # create history list
        self.history = []
        self.legalMoves = []

        # initialize players
        self.player_current  = BLACK
        self.player_opponent = WHITE

        # create arrays and lists
        # +1 on board size is used as an border location used for all borders
        self.board      = <char  *>malloc( ( size * size + 1 ) * sizeof( char  ) )
        self.hash3x3    = <long  *>malloc( ( size * size     ) * sizeof( long  ) )
        self.boardGroups = []
        self.groupsList  = []

        # create empty location group
        cdef Group boardGroup    = Group()

        # initialize board
        for i in range( size * size ):
            self.board[   i ]     = EMPTY
            self.hash3x3[ i ]     = 0
            self.boardGroups.append( boardGroup )
            self.legalMoves.append( i )

        # initialize border location
        self.board[ ( size * size ) ] = BORDER
        self.boardGroups.append( boardGroup )

        # initialize neighbor locations
        self.set_neighbors(     size )
        self.set_3x3_neighbors( size )
        self.set_12d_neighbors( size )

        # initialize zobrist hash
        # TODO optimize?
        rng = np.random.RandomState(0)
        self.hash_lookup = {
            WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
            BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        self.current_hash = np.uint64(0)
        self.previous_hashes = set()


    # duplicate given Gamestate
    cdef initialize_duplicate( self, GameState copyState ):

        # !!! do not copy !!! -> these do not need a deep copy as they are static
        self.neighbor    = copyState.neighbor
        self.neighbor3x3 = copyState.neighbor3x3
        self.neighbor12d = copyState.neighbor12d

        ########################################################################
        # TODO deep copy all other data
        self.size  = copyState.size
        self.board = copyState.board


    def __init__( self, char size = 19, GameState copyState = None ):
        # create new instance of GameState

        if copyState is None:

            # create root state with empty board
            # and initialize all arrays
            print( "init root state" )
            self.initialize_new( size )

        else:

            # create copy of given state
            print( "copy state" )
            self.initialize_duplicate( copyState )


    # deallocate all arrays
    # arrays created with malloc have to be freed when this instance is destroyed
    def __dealloc__(self):

        if self.board is not NULL:
            free( self.board )

        if self.hash3x3 is not NULL:
            free( self.hash3x3 )

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    # combine two groups and remove one
    cdef void combine_groups( self, Group group_keep, Group group_remove ):

        # remove group_remove from groupList
        self.groupsList.remove( group_remove )

        # combine stones, liberty
        # TODO speed gain, determine witch one is bigger? and add small one to big one
        group_keep.location_liberty.update( group_remove.location_liberty )
        group_keep.location_stones.update(  group_remove.location_stones  )

        # set all locations of group_remove to group_keep
        cdef short location
        for location in group_remove.location_stones:

            self.boardGroups[ location ] = group_keep        


    # remove group from board
    cdef void remove_group( self, Group group_remove ):

        cdef short location
        cdef short neighbor_location
        cdef Group group_temp
        cdef char  board_value
        cdef int   i
        # empty group is always in border location
        cdef Group group_empty = self.boardGroups[ self.size * self.size ]

        self.groupsList.remove( group_remove )

        # loop over all group stone locations
        for location in group_remove.location_stones:

            # set location to empty group
            self.boardGroups[ location ] = group_empty

            # set boardLocation to empty
            self.board[ location ] = EMPTY
            self.legalMoves.append( location )

            # update liberty of neighbors
            # loop over all four neighbors
            for i in range( 4 ):

                # get neighbor location
                neighbor_location = self.neighbor[ location * 4 + i ]

                # only current_player groups can be next to a killed group
                # check if there is a group
                board_value = self.board[ neighbor_location ]
                if board_value == self.player_current:

                    # add liberty
                    group_temp = self.boardGroups[ neighbor_location ]
                    group_temp.location_liberty[ location ] = location

        # update all 3x3 hashes in update_hash_locations

    # add location to group or create new one
    cdef void add_to_group( self, short location ):

        cdef Group newGroup = None
        cdef Group tempGroup
        cdef short neighborLocation
        cdef char  boardValue
        cdef int   i

        # find friendly stones
        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location and value
            neighborLocation = self.neighbor[ location * 4 + i ]
            boardValue       = self.board[ neighborLocation ]

            # check if neighbor is friendly stone
            if boardValue == self.player_current:

                # check if this is the first friendly neighbor we found
                if newGroup is None:

                    # first friendly neighbor
                    newGroup = self.boardGroups[ neighborLocation ]
                else:

                    # another friendly group, if they are different combine them
                    tempGroup = self.boardGroups[ neighborLocation ]
                    if tempGroup != newGroup:
                        self.combine_groups( newGroup, tempGroup )

            elif boardValue == self.player_opponent:

                # remove liberty from enemy group
                tempGroup = self.boardGroups[ neighborLocation ]
                tempGroup.location_liberty.pop( location, None )
                # kill group
                if len( tempGroup.location_liberty ) == 0:

                    self.remove_group( tempGroup )

        # check if a group was found or create one
        if newGroup is None:    

            newGroup = Group()
            self.groupsList.append( newGroup )
        else:

            newGroup.location_liberty.pop( location, None )

        # add stone and remove liberty
        newGroup.location_stones[ location ] = location

        # add new liberty
        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighborLocation = self.neighbor[ location * 4 + i ]

            if self.board[ neighborLocation ] == EMPTY:
                newGroup.location_liberty[ neighborLocation ] = neighborLocation
 
        # set location group

        self.boardGroups[ location ] = newGroup

    # generate 12d hash around location
    cdef long generate_12d_hash( self, short centre ):
        # return 12d pattern hash

        cdef int i
        cdef long hash = HASHVALUE

        # hash all stone locations
        for i in range( 12 ):
            hash += self.board[ self.neighbor12d[ centre * 12 + i ] ]
            hash *= HASHVALUE

        # hash all liberty locations
        for i in range( 12 ):
            hash += self.liberties[ self.neighbor12d[ centre * 12 + i ] ]
            hash *= HASHVALUE

        return hash

    # generate 3x3 hash around location
    cdef long generate_3x3_hash( self, short centre ):
        # return 3x3 pattern hash

        cdef int i
        cdef long hash = HASHVALUE

        # hash all stone locations
        for i in range( 8 ):
            hash += self.board[ self.neighbor3x3[ centre * 8 + i ] ]
            hash *= HASHVALUE

        # hash all liberty locations
        for i in range( 8 ):
            hash += self.liberties[ self.neighbor3x3[ centre * 8 + i ] ]
            hash *= HASHVALUE

        return hash

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # return hash for 12d star pattern around location
    cdef long get_hash_12d( self, short centre ):
        # return 12d pattern hash

        return self.generate_12d_hash( centre )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # return 3x3 pattern hash + current player
    cdef long get_hash_3x3( self, short location ):
        # return 3x3 pattern hash

        # 3x3 hash patterns are updated every move
        # get 3x3 hash value and add current player 
        return self.hash3x3[ location ] + self.player_current

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # return value relative to current player
    cdef char get_board_feature( self, short location ):
        # return correct board feature value
        # 0 active player stone
        # 1 opponent stone
        # 2 empty location

        cdef char value = self.board[ location ]

        if value == EMPTY:
            return 2

        if value == self.player_current:
            return 0

        return 1

    # return 1 if ladder capture, 0 if not
    cdef char is_ladder_capture( self, short location ):
        #
        return 0

    # return 1 if ladder escape, 0 if not
    cdef char is_ladder_escape( self, short location ):
        #
        return 0

    cdef short get_move_history( self ):
        #

        return 0

    cdef short get_liberties( self, short location, short max ):
        #

        cdef short value = self.liberties[ location ]

        if max > value:
            return value

        return max

    cdef short get_liberties_after( self, short location, short max ):
        #

        return 0

    cdef short get_capture( self, short location, short max ):
        #

        return 0

    cdef short get_self_atari_size( self, short location, short max ):
        #

        return 0

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    # TODO
    # play move and update liberties hashes etc.
    cdef void add_move( self, short location ):

        # assume legal move, or check it?

        self.board[ location ] = self.player_current
        self.add_to_group( location )

        # change player colour
        self.player_current = self.player_opponent
        self.player_opponent = ( BLACK if self.player_current == WHITE else WHITE )

        # add to history
        self.history.append( location )

        # generate legal moves? -> or should be done when board changes

        self.legalMoves.remove( location )

        # update zobrist

    # copy gamestate (self) and play move
    cdef GameState new_state_add_move( self, short location ):
        # create new state including all data of self and add move

        # create new gamestate, copy all data of self
        state = GameState( copyState = self )

        # place move
        state.add_move( location )

        return state


    # return winner colour
    cdef char get_winner_colour( self, char komi ):
        """
           Calculate score of board state and return player ID ( BLACK, WHITE, or EMPTY for tie )
           corresponding to winner. Uses 'Area scoring'.
        """

        return 0

    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################

    # do move, throw exception when outside the board
    # action has to be a ( x, y ) tuple
    # this function should be used from Python environment, 
    # use add_move from C environment for speed
    def do_move( self, action ):

        # do move, return true if legal, return false if not

        cdef int x, y
        ( x, y ) = action
        self.add_move( y + ( x * self.size ) )


    # copy state and play move
    # action has to be a ( x, y ) tuple
    # this function should be used from Python environment, 
    # use new_state_add_move from C environment for speed
    def get_next_state( self, action ):

        # calculate location from tuple action
        cdef char x, y
        ( x, y ) = action
        cdef short location
        location = x + ( y * self.size )

        return self.new_state_add_move( location )

    # add handicap stones
    def place_handicap( self, handicap ):
        # add handicap stones
        # list with tuples black stones will be added accordingly

        return 0

    # return winner colour
    def get_winner( self, char komi ):
        """
           Calculate score of board state and return player ID ( 1, -1, or 0 for tie )
           corresponding to winner. Uses 'Area scoring'.
        """

        return self.get_winner_colour( komi )

    # return list with legal moves
    def get_legal_moves( self, include_eyes = True ):
        # return all legal moves, in/excluding eyes

        return 0

    # return true/false if move at action is legal
    def is_legal( self, action ):

        cdef char x, y
        cdef short loc
        ( x, y ) = action

        # check outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # calculate location
        loc = x + ( y * self.size )

        return True

    # copy this state
    def copy( self ):

        return GameState( copyState = self )

    def reset( self ):

        cdef short i
        cdef int size = self.size
        # create history list
        self.history = []
        self.legalMoves = []

        # initialize players
        self.player_current  = BLACK
        self.player_opponent = WHITE

        # create arrays and lists
        # +1 on board size is used as an border location used for all borders
        self.board      = <char  *>malloc( ( size * size + 1 ) * sizeof( char  ) )
        self.hash3x3    = <long  *>malloc( ( size * size     ) * sizeof( long  ) )
        self.boardGroups = []
        self.groupsList  = []

        # create empty location group
        cdef Group boardGroup    = Group()

        # initialize board
        for i in range( size * size ):
            self.board[   i ]     = EMPTY
            self.hash3x3[ i ]     = 0
            self.boardGroups.append( boardGroup )
            self.legalMoves.append( i )

        # initialize border location
        self.board[ ( size * size ) ] = BORDER
        self.boardGroups.append( boardGroup )

        # initialize zobrist hash
        # TODO optimize?
        rng = np.random.RandomState(0)
        self.hash_lookup = {
            WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
            BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        self.current_hash = np.uint64(0)
        self.previous_hashes = set()

    ############################################################################
    #   tests                                                                  #
    #                                                                          #
    ############################################################################

    def printer( self ):
        for i in range( self.size ):
            A = str( i ) + " "
            for j in range( self.size ):
                A += str( self.board[ j + i * self.size ] ) + " "
            print( A )

    cdef test( self ):
        cdef int i
        prep  = Preprocess( [ "board" ] )
        for i in range( 100 ):
            prep.state_to_tensor( self )

    def test_speed( self ):

        start = time.time()
        cdef int x
        cdef GameState state

        prep  = Preprocess( [ "board" ] )
        # state = GameState( copyState = self )
        # state.do_move( (10, 10) )

        while x < 10000:
            prep.state_to_tensor( self )
            x += 1

        print( "time " + str( time.time() - start ) )

