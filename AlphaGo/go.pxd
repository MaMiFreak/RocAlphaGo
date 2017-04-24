import numpy as np
cimport numpy as np

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
    cdef void set_neighbors( self, int size )
    cdef void set_3x3_neighbors(self, int size)
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
    cdef long generate_12d_hash( self, short centre )
    cdef long generate_3x3_hash( self, short centre )

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    cdef long get_hash_12d( self, short centre )
    cdef long get_hash_3x3( self, short location )
    cdef char get_board_feature( self, short location )
    cdef char is_ladder_capture( self, short location )
    cdef char is_ladder_escape( self, short location )
    cdef short get_move_history( self )
    cdef short get_liberties( self, short location, short max )
    cdef short get_liberties_after( self, short location, short max )
    cdef short get_capture( self, short location, short max )
    cdef short get_self_atari_size( self, short location, short max )

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    cdef void add_move( self, short location )
    cdef GameState new_state_add_move( self, short location )
    cdef char get_winner_colour( self, char komi )

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
