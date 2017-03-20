import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

PASS_MOVE = -1  # have to check if this works

cdef char EMPTY  = 0
cdef char WHITE  = 1
cdef char BLACK  = 2
cdef char BORDER = 4

# value used to generate pattern hashes
cdef char HASHVALUE = 33


# structure to store group information
cdef class Group:
    cdef dict location_stones
    cdef dict location_liberty

# structure to store ladder group information
cdef class LadderGroup:
    cdef dict location_stones
    cdef dict location_liberty
    cdef dict escape_atari    

cdef class GameState:

    # private variables


    """ -> public variables, declared in go.pxd

    cdef object legalMoves

    # general game info
    cdef public char size

    # board state info
    cdef short ko       # possible ko location
    cdef short *board
    cdef Group *groups

    # opponent & active player colour
    cdef char player_current
    cdef char player_opponent

    cdef object history

    # keep track of 3x3 pattern hashes
    cdef long  *hash3x3

    # neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef dict   hash_lookup
    cdef object current_hash
    cdef object previous_hashes
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
    cdef set_root( self, char size ):

        cdef int i
        self.size = size
        self.history = []

        # initilize players
        self.player_current  = BLACK
        self.player_opponent = WHITE

        # initialize numpy board
        self.boardNumpy = np.zeros( size * size )

        # initialize board
        for i in range(361):
            self.board[i]     = EMPTY
            self.groups[i]    = 0
            self.hash3x3[i]   = 0
            self.liberties[i] = 0

        # initialize border
        self.board[ 361 ]     = BORDER
        self.liberties[ 361 ] = 0

        # initialize neighbor locations
        self.set_neighbors(     size )
        self.set_3x3_neighbors( size )
        self.set_12d_neighbors( size )

        # zobrist hash
        rng = np.random.RandomState(0)
        self.hash_lookup = {
            WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
            BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        self.current_hash = np.uint64(0)
        self.previous_hashes = set()


    # duplicate given Gamestate
    cdef set_duplicate( self, GameState copyState ):

        # !!! do not copy !!! -> these do not need a deep copy as they are static
        self.neighbor    = copyState.neighbor
        self.neighbor3x3 = copyState.neighbor3x3
        self.neighbor12d = copyState.neighbor12d

        # deep copy all other data
        self.size  = copyState.size
        self.board = copyState.board


    def __init__( self, char size = 19, GameState copyState = None ):

        # create new instance of GameState

        if copyState is None:

            # create root state with empty board
            # and initialize all arrays
            print( "init root state" )
            self.set_root( size )

        else:

            # create copy of given state
            print( "copy state" )
            self.set_duplicate( copyState )


    ############################################################################
    #   private cdef functions                                                 #
    #                                                                          #
    ############################################################################

    # map group at location with same colour
    cdef dict map_group( self, short location ):
        cdef dict group
        cdef dict locations

        group = { location : location }
        locations = { location : location }

        return group


    # generate hash for 12d start pattern around location
    cdef long get_hash_12d_response( self, char x, char y, short centre ):
        # return 12d pattern hash

        cdef int i
        cdef long hash
        hash = HASHVALUE

        # hash all stone locations
        for i in range( 12 ):
            hash += self.board[ self.neighbor12d[ centre * 12 + i ] ]
            hash *= HASHVALUE

        # hash all liberty locations
        for i in range( 12 ):
            hash += self.liberties[ self.neighbor12d[ centre * 12 + i ] ]
            hash *= HASHVALUE

        return hash

    # generate hash for 3x3 start pattern around location
    cdef long get_hash_3x3_non_response( self, short location ):
        # return 3x3 pattern hash

        return self.hash3x3[ location ] + self.player_current

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

    cdef char is_ladder_capture( self, short location ):
        #
        return 0

    cdef char is_ladder_escape( self, short location ):
        #
        return 0

    cdef short get_stone_age( self, short location, short max ):
        #

        return max

    cdef short get_liberties( self, short location, short max ):
        #

        cdef short value = self.liberties[ location ]

        if max > value:
            return value

        return max

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    # play move and update liberties hashes etc.
    cdef void add_move( self, short location ):
        # place move and update all hash/liberty etc.

        self.board[ location ] = self.player_current
        self.boardNumpy[ location ] = self.player_current

        # change player colour
        self.player_current = self.player_opponent
        self.player_opponent = ( BLACK if self.player_opponent == WHITE else WHITE )

    # copy gamestate and play move
    cdef GameState new_state_add_move( self, short location ):
        # create new state including all data and add move
        # return new state

        # create new gamestate, copy all data
        state = GameState( copyState = self )

        # place move
        state.add_move( location )

        return state

    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################

    # do move, throw exception when outside the board
    def do_move( self, action ):
        # do move, return true if legal, return false if not

        cdef int x, y
        ( x, y ) = action
        self.add_move( x + ( y * self.size ) )

    # copy state and play move
    def get_next_state( self, action ):
        # create new state including all settings and do move
        # return new state

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
    def get_winner( self, komi ):
        """Calculate score of board state and return player ID ( 1, -1, or 0 for tie )
        corresponding to winner. Uses 'Area scoring'.
        """

        return 0

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

    def test( self, location ):
        for i in range( 4 ):
            print( "neighbor " + str( self.neighbor[ location * 4 + i ] ) )

    def test_speed( self ):

        import time
        from AlphaGo.preprocessing.preprocessing import Preprocess

        start = time.time()
        cdef int x
        cdef GameState state

        prep  = Preprocess( [ "board" ] )
        # state = GameState( copyState = self )
        # state.do_move( (10, 10) )

        for x in range(100000):
            prep.state_to_tensor( self )

        print( "time " + str( time.time() - start ) )
