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
