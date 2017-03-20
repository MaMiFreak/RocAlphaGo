import numpy as np
cimport numpy as np

cdef class GameState:

    # general game info
    cdef char size

    # right now fixed 19 * 19 board size
    # TODO make board size variable -> pointers!
    # board state info
    cdef short ko                # possible ko location
    cdef short liberties[ 362 ]  # +1 for border location
    cdef char      board[ 362 ]  # +1 for border location
    cdef np.ndarray boardNumpy
    cdef char     groups[ 361 ]  # exact implementation? group struct? int? memory vs speed

    cdef char player_current
    cdef char player_opponent

    cdef object history

    # keep track of 3x3 pattern hashes
    cdef long hash3x3[ 361 ]

    # neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef dict   hash_lookup
    cdef object current_hash
    cdef object previous_hashes
    cdef object legalMoves

    cdef set_root( self, char size )
    cdef set_duplicate( self, GameState copyState )
    cdef char get_board_feature( self, short location )
    cdef void add_move( self, short location )

