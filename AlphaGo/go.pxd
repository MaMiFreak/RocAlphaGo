import numpy as np
cimport numpy as np

# observe stones > EMPTY
#         border < EMPTY
# be aware you should NOT use != EMPTY as this includes border locations
cdef char PASS   = -1
cdef char BORDER = 0
cdef char EMPTY  = 1
cdef char WHITE  = 2
cdef char BLACK  = 3

# structure to store ladder group information
cdef class LadderGroup:
    cdef dict location_stones
    cdef dict location_liberty
    cdef dict escape_atari  

# structure to store group information
cdef class Group:
    cdef dict location_stones
    cdef dict location_liberty

cdef class GameState:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

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
    cdef dict   hash_lookup
    cdef int    current_hash
    cdef set    previous_hashes

    cdef list   legalMoves

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################
    cdef short calculate_board_location( self, char x, char y )
    cdef short calculate_board_location_or_border( self, char x, char y )
    cdef void set_neighbors( self, int size )
    cdef void set_3x3_neighbors(self, int size)
    cdef void set_3x3_neighbors_backup(self, int size)
    cdef void set_12d_neighbors( self, int size )
    cdef initialize_new( self, char size )
    cdef initialize_duplicate( self, GameState copyState )

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    cdef void combine_groups( self, Group group_keep, Group group_remove )
    cdef void remove_group( self, Group group_remove )
    cdef void add_to_group( self, short location )
    cdef bint has_liberty_after( self, short location )
    cdef bint is_legal_move( self, short location )
    cdef long generate_12d_hash( self, short centre )
    cdef long generate_3x3_hash( self, short centre )
    cdef Group get_group_after( self, short location )
    cdef bint is_ladder_capture_move( self, GameState state, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase )
    cdef bint is_ladder_escape_move( self, GameState state, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase )
    cdef char get_winner_colour( self, int komi )

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    cdef long get_hash_12d( self, short centre )
    cdef long get_hash_3x3( self, short location )
    cdef char get_board_feature( self, short location )
    cdef list get_groups_after( self )
    cdef char is_ladder_capture( self, Group group, short location, int maxDepth )
    cdef char is_ladder_escape( self, Group group, short location, int maxDepth )
    cdef bint is_true_eye( self, short location, dict eyes, char owner )
    cdef list get_sensible_moves( self )
    cdef short get_capture_size( self, short location, short max )

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    cdef void add_move( self, short location )
    cdef GameState new_state_add_move( self, short location )

    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################

    #def do_move( self, action )
    #def get_next_state( self, action )
    #def place_handicap( self, handicap )
    #def get_winner( self, char komi )
    #def get_legal_moves( self, include_eyes = True )
    #def is_legal( self, action )

    ############################################################################
    #   tests                                                                  #
    #                                                                          #
    ############################################################################

    #def printer( self )
    cdef test( self )
    #def test_speed( self )
