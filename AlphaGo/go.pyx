import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython
import time
from AlphaGo.preprocessing.preprocessing import Preprocess

# observe stones > EMPTY
#         border < EMPTY
# be aware you should NOT use != EMPTY as this includes border locations
cdef char PASS   = -1
cdef char BORDER = 0
cdef char EMPTY  = 1
cdef char WHITE  = 2
cdef char BLACK  = 3

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


    # return location on board
    # no checks on outside board
    # x = columns
    # y = rows
    cdef short calculate_board_location( self, char x, char y ):

        # return board location
        return x + ( y * self.size )

    # return location on board or borderlocation
    # board locations = [ 0, size * size )
    # border location = size * size
    # x = columns
    # y = rows
    cdef short calculate_board_location_or_border( self, char x, char y ):

        # check if x or y are outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            # return border location
            return self.size * self.size

        # return board location
        return self.calculate_board_location( x, y )


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
                self.neighbor[ location + 0 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor[ location + 1 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor[ location + 2 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor[ location + 3 ] = self.calculate_board_location_or_border( x    , y + 1 )


    # create for every board location array with all 8 surrounding neighbour locations
    # neighbor order: above middle - middle left - middle right - below middle
    #                 above left - above right - below left - below right
    #                 this order is more useful as it separates neighbors and then diagonals
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
                self.neighbor3x3[ location + 0 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor3x3[ location + 1 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor3x3[ location + 2 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor3x3[ location + 3 ] = self.calculate_board_location_or_border( x    , y + 1 )

                self.neighbor3x3[ location + 4 ] = self.calculate_board_location_or_border( x - 1, y - 1 )
                self.neighbor3x3[ location + 5 ] = self.calculate_board_location_or_border( x + 1, y - 1 )
                self.neighbor3x3[ location + 6 ] = self.calculate_board_location_or_border( x - 1, y + 1 )
                self.neighbor3x3[ location + 7 ] = self.calculate_board_location_or_border( x + 1, y + 1 )


    # create for every board location array with all 8 surrounding neighbour locations
    # neighbor order: above left - above middle - above right
    #                 left - right
    #                 below left - below middle - below right
    # -1    xxx
    #       x x
    # +1    xxx
    #
    cdef void set_3x3_neighbors_backup(self, int size):

        # create array
        self.neighbor3x3 = <short *>malloc( size * size * 8 * sizeof( short ) )
        cdef short location
        cdef char x, y

        # add all surrounding neighbors to every board location
        for x in range( size ):
            for y in range( size ):
                location = ( x + ( y * size ) ) * 8
                self.neighbor3x3[ location + 0 ] = self.calculate_board_location_or_border( x - 1, y - 1 )
                self.neighbor3x3[ location + 1 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor3x3[ location + 2 ] = self.calculate_board_location_or_border( x + 1, y - 1 )
                self.neighbor3x3[ location + 3 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor3x3[ location + 4 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor3x3[ location + 5 ] = self.calculate_board_location_or_border( x - 1, y + 1 )
                self.neighbor3x3[ location + 6 ] = self.calculate_board_location_or_border( x    , y + 1 )
                self.neighbor3x3[ location + 7 ] = self.calculate_board_location_or_border( x + 1, y + 1 )


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
                self.neighbor12d[ location +  4 ] = self.calculate_board_location_or_border( x    , y - 2 )

                self.neighbor12d[ location +  1 ] = self.calculate_board_location_or_border( x - 1, y - 1 )
                self.neighbor12d[ location +  5 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor12d[ location +  8 ] = self.calculate_board_location_or_border( x + 1, y - 1 )

                self.neighbor12d[ location +  0 ] = self.calculate_board_location_or_border( x - 2, y     )
                self.neighbor12d[ location +  2 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor12d[ location +  9 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor12d[ location + 11 ] = self.calculate_board_location_or_border( x + 2, y     )

                self.neighbor12d[ location +  3 ] = self.calculate_board_location_or_border( x - 1, y + 1 )
                self.neighbor12d[ location +  6 ] = self.calculate_board_location_or_border( x    , y + 1 )
                self.neighbor12d[ location + 10 ] = self.calculate_board_location_or_border( x + 1, y + 1 )

                self.neighbor12d[ location +  7 ] = self.calculate_board_location_or_border( x    , y + 2 )


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
        # rng = np.random.RandomState(0)
        # self.hash_lookup = {
        #    WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
        #    BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        # self.current_hash = np.uint64(0)
        # self.previous_hashes = set()


    # duplicate given Gamestate
    cdef initialize_duplicate( self, GameState copy_state ):

        cdef int   i
        cdef short location
        cdef Group group, group_new

        # !!! do not copy !!! -> these do not need a deep copy as they are static
        self.neighbor        = copy_state.neighbor
        self.neighbor3x3     = copy_state.neighbor3x3
        self.neighbor12d     = copy_state.neighbor12d

        # pattern dictionary

        # zobrist
      # self.hash_lookup     = copy_state.hash_lookup

        ########################################################################
        # TODO deep copy all other data

        # set values
        self.ko              = copy_state.ko
        self.size            = copy_state.size
        self.player_current  = copy_state.player_current
        self.player_opponent = copy_state.player_opponent
      # self.current_hash    = copy_state.current_hash

        # copy values
        self.history         = list( copy_state.history )
        self.legalMoves      = list( copy_state.legalMoves )
      # self.previous_hashes = list( copy_state.previous_hashes )

        # create array/list
        self.board           = <char  *>malloc( ( self.size * self.size + 1 ) * sizeof( char  ) )
        self.hash3x3         = <long  *>malloc( ( self.size * self.size     ) * sizeof( long  ) )
        self.boardGroups     = []
        self.groupsList      = []

        # border group does not need to be a copy
        group = copy_state.boardGroups[ self.size * self.size ]

        # copy array/list values
        for i in range( self.size * self.size ):
            self.board[   i ]     = copy_state.board[   i ]
            self.hash3x3[ i ]     = copy_state.hash3x3[ i ]
            self.boardGroups.append( group )

        # initialize border location
        self.board[ ( self.size * self.size ) ] = BORDER
        self.boardGroups.append( group )

        # copy all groups and set groupsList
        for group in copy_state.groupsList:
            group_temp = Group()
            group_temp.location_stones  = group.location_stones.copy()
            group_temp.location_liberty = group.location_liberty.copy()

            self.groupsList.append( group_temp )

            # loop over all group locations and set group
            for location in group_temp.location_stones:

                self.boardGroups[ location ] = group_temp

    def __init__( self, char size = 19, GameState copyState = None ):
        # create new instance of GameState

        if copyState is None:

            # create root state with empty board
            # and initialize all arrays
            # print( "init root state" )
            self.initialize_new( size )

        else:

            # create copy of given state
            # print( "copy state" )
            self.initialize_duplicate( copyState )


    # deallocate all arrays
    # arrays created with malloc have to be freed when this instance is destroyed
    def __dealloc__(self):

        #if self.board is not NULL:
        free( self.board )

        #if self.hash3x3 is not NULL:
        free( self.hash3x3 )

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    # combine two groups and remove one
    @cython.boundscheck( False )
    @cython.wraparound(  False )
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
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void remove_group( self, Group group_remove ):

        cdef short location
        cdef short neighbor_location
        cdef Group group_temp
        cdef char  board_value
        cdef int   i
        # empty group is always in border location
        cdef Group group_empty = self.boardGroups[ self.size * self.size ]

        # remove group_remove from groupList
        self.groupsList.remove( group_remove )

        # if groupsize == 1, possible ko
        if len( group_remove.location_stones ) == 1:
            self.ko = group_remove.location_stones.values()[0]

        # loop over all group stone locations
        for location in group_remove.location_stones:

            # set location to empty group
            self.boardGroups[ location ] = group_empty

            # set boardLocation to empty
            self.board[ location ] = EMPTY

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
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void add_to_group( self, short location ):

        cdef Group newGroup = None
        cdef Group tempGroup
        cdef short neighborLocation
        cdef char  boardValue
        cdef char  group_removed = 0
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

                # remove group
                if len( tempGroup.location_liberty ) == 0:

                    self.remove_group( tempGroup )
                    group_removed += 1

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
 
        # if two groups died there is no ko
        if group_removed >= 2:
             self.ko = PASS

        # set location group
        self.boardGroups[ location ] = newGroup

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint has_liberty_after( self, short location ):
        """
           check if a play at location results in an alive group
           - has liberty
           - conects to group with >= 2 liberty
           - captures enemy group
        """
        cdef int   i
        cdef char  board_value
        cdef short neighbor_location
        cdef Group group_temp

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighbor_location = self.neighbor[ location * 4 + i ]
            board_value       = self.board[ neighbor_location ]

            # check empty location -> liberty
            if board_value == EMPTY:

                return 1

            # get neighbor group
            group_temp = self.boardGroups[ neighbor_location ]

            # if there is a player_current group
            if board_value == self.player_current:
                
                # if it has at least 2 liberty
                if len( group_temp.location_liberty ) >= 2:
                    return 1
            # if is a player_opponent group and has only onle liberty
            elif board_value == self.player_opponent and len( group_temp.location_liberty ) == 1:
                return 1
            
        return 0
    
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_legal_move( self, short location ):
        """
           check if playing at location is a legal move to make
        """

        # check if it is empty
        if self.board[ location ] != EMPTY:
            return 0

        # check ko
        if location == self.ko:
            return 0

        # check if it has liberty after
        if not self.has_liberty_after( location ):
            return 0
        # super-ko

        return 1

    # generate 12d hash around centre location
    @cython.boundscheck( False )
    @cython.wraparound(  False )
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

    # generate 3x3 hash around centre location
    @cython.boundscheck( False )
    @cython.wraparound(  False )
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

    # return group as it is after playing at location
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef Group get_group_after( self, short location ):

        cdef Group new_group = Group()
        cdef short neighbor_location
        cdef short temp_location
        cdef char  board_value
        cdef Group temp_group
        cdef int   i
        cdef dict  stones_removed = {}

        # find friendly stones
        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location and value
            neighbor_location = self.neighbor[ location * 4 + i  ]
            board_value       = self.board[    neighbor_location ]

            # check if neighbor is friendly stone
            if board_value == EMPTY:

                new_group.location_liberty[ neighbor_location ] = neighbor_location
            elif board_value == self.player_current:

                # found friendly group
                temp_group = self.boardGroups[ neighbor_location ]

                new_group.location_liberty.update( temp_group.location_liberty )
                new_group.location_stones.update(  temp_group.location_stones  )

            elif board_value == self.player_opponent:

                # get enemy group
                temp_group = self.boardGroups[ neighbor_location ]

                # if it has one liberty it wil be killed -> add potential liberty
                if len( temp_group.location_liberty ) == 1:

                    stones_removed.update( temp_group.location_stones )

        # add stone
        new_group.location_stones[ location ] = location

        for neighbor_location in stones_removed:
            # loop over all four neighbors
            for i in range( 4 ):

                # get neighbor location and value
                temp_location = self.neighbor[ neighbor_location * 4 + i ]
                if temp_location in new_group.location_stones:

                    new_group.location_liberty[ neighbor_location ] = neighbor_location

        # remove location as liberty
        new_group.location_liberty.pop( location, None )

        return new_group

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    # return hash for 12d star pattern around location
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long get_hash_12d( self, short centre ):
        # return 12d pattern hash

        return self.generate_12d_hash( centre )

    # return 3x3 pattern hash + current player
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long get_hash_3x3( self, short location ):
        # return 3x3 pattern hash

        # 3x3 hash patterns are updated every move
        # get 3x3 hash value and add current player 
        return self.hash3x3[ location ] + self.player_current

    # return value relative to current player
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char get_board_feature( self, short location ):
        """
           return correct board feature value
           - 0 active player stone
           - 1 opponent stone
           - 2 empty location
        """

        cdef char value = self.board[ location ]

        if value == EMPTY:
            return 2

        if value == self.player_current:
            return 0

        return 1

    cdef bint is_group_in_ladder( self, char  *board, LadderGroup group, short location, int maxDepth, char group_colour, char chase_colour ):

        cdef short location_liberty
        cdef LadderGroup group_temp

        # if we haven't found a capture by a certain number of moves, assume it's worked.
        if maxDepth <= 0:

            return 1

        # place stone
        board[ location ] = chase_colour

        # remove liberty
        group.location_liberty.pop( location )

        for location_liberty in group.location_liberty:

            group_temp                  = LadderGroup()
            group_temp.escape_atari     = group.escape_atari.copy()
            group_temp.location_stones  = group.location_stones.copy()
            group_temp.location_liberty = group.location_liberty.copy()

            # if one liberty is an escape -> group can escape
            if self.can_group_escape_ladder( board, group_temp, location_liberty, maxDepth - 1, group_colour, chase_colour ):

                # no ladder capture
                return 0

            # some stones are in atari, kill one of them in order to get more liberty
        

        # no ladder escape found -> group is captured
        return 1

    cdef bint can_group_escape_ladder( self, char  *board, LadderGroup group, short location, int maxDepth, char group_colour, char chase_colour ):

        cdef int   i
        cdef Group group_temp
        cdef LadderGroup ladder_group_temp
        cdef char  board_value
        cdef short location_neighbor

        # if we haven't found an escape by a certain number of moves, give up.
        if maxDepth <= 0:

            return 0

        # place stone
        board[ location ] = group_colour

        # add stone to group
        group.location_stones[ location ] = location

        # loop over nieghbor
        for i in range( 4 ):

            location_neighbor = self.neighbor[ location * 4 + i ]
            board_value       = board[ location_neighbor ]

            if board_value == EMPTY:

                # add new liberty
                group.location_liberty[ location_neighbor ] = location_neighbor

            if board_value == group_colour:

                # friendly group -> add stones and liberty
                group_temp = self.boardGroups[ location_neighbor ]
                group.location_stones.update( group_temp.location_stones )
                group.location_liberty.update( group_temp.location_liberty )

        # remove liberty group
        group.location_liberty.pop( location )

        i = len( group.location_liberty )
        # if less than 2 liberty -> capture
        if i < 2:

            # no escape
            return 0

        # if more than 2 liberty -> escape
        elif i > 2:

            # escaped
            return 1

        # 2 liberty -> still ladder, need iterative check

        # we have two liberty, if both are not a capture this is a escape
        for location_neighbor in group.location_liberty:
            
            ladder_group_temp                  = LadderGroup()
            ladder_group_temp.escape_atari     = group.escape_atari.copy()
            ladder_group_temp.location_stones  = group.location_stones.copy()
            ladder_group_temp.location_liberty = group.location_liberty.copy()

            if self.is_group_in_ladder( board, ladder_group_temp, location_neighbor, maxDepth - 1, group_colour, chase_colour ):

                # no escape
                return 0
            # undo last move
            board[ location_neighbor ] = EMPTY

        # escaped
        return 1

    # return 1 if ladder capture, 0 if not
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char is_ladder_capture( self, Group group, short location, int maxDepth ):
        #

        cdef int i
        cdef short location_temp
        cdef LadderGroup ladder_group = LadderGroup()
        cdef char  *board = <char  *>malloc( ( self.size * self.size + 1 ) * sizeof( char  ) )

        # create ladderGroup to escape
        ladder_group.location_stones  = group.location_stones.copy()
        ladder_group.location_liberty = group.location_liberty.copy()

        # find all groups in atary

        # duplicate board
        for i in range( self.size * self.size + 1 ):
            board[ i ] = self.board[ i ]

        # try to escape ladder
        if self.is_group_in_ladder( board, ladder_group, location, maxDepth, self.player_opponent, self.player_current ):
            return 1

        return 0

    # return 1 if ladder escape, 0 if not
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char is_ladder_escape( self, Group group, short location, int maxDepth ):
        #

        cdef int i
        cdef short location_temp
        cdef LadderGroup ladder_group = LadderGroup()
        cdef char  *board = <char  *>malloc( ( self.size * self.size + 1 ) * sizeof( char  ) )

        # create ladderGroup to escape
        ladder_group.location_stones  = group.location_stones.copy()
        ladder_group.location_liberty = group.location_liberty.copy()

        # find all groups in atary

        # duplicate board
        for i in range( self.size * self.size + 1 ):
            board[ i ] = self.board[ i ]

        # try to escape ladder
        if self.can_group_escape_ladder( board, ladder_group, location, maxDepth, self.player_current, self.player_opponent ):
            return 1

        return 0

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_true_eye( self, short location, dict eyes ):

        cdef int   i
        cdef char  board_value
        cdef char  max_bad_diagonal
        cdef char  count_bad_diagonal = 0
        cdef char  count_border = 0
        cdef short location_neighbor
        cdef list  empty_diagonal = []
        cdef dict  future_eyes
        
        # TODO benchmark what is faster? first dict lookup then neighbor check or other way around

        # check if it is a known eye
        if location in eyes:

            return 1

        # loop over neighbor
        for i in range( 4 ):

            location_neighbor = self.neighbor3x3[ location * 8 + i ]
            board_value       = self.board[ location_neighbor ]

            if board_value == BORDER:

                count_border += 1
            elif board_value != self.player_current:

                # empty location or enemy stone
                return 0

        # loop over diagonals
        for i in range( 4, 8 ):

            location_neighbor = self.neighbor3x3[ location * 8 + i ]
            board_value       = self.board[ location_neighbor ]

            if board_value == EMPTY:

                empty_diagonal.append( location_neighbor )
                count_bad_diagonal += 1
            elif board_value == BORDER:

                count_border += 1
            elif board_value == self.player_opponent:

                # enemy stone
                count_bad_diagonal += 1

        # assume location is an eye
        future_eyes = eyes.copy()
        future_eyes[ location ] = location 

        max_bad_diagonal = 1 if count_border == 0 else 0

        if count_bad_diagonal <= max_bad_diagonal:

            # one bad diagonal is allowed in the middle
            return 1

        for location_neighbor in empty_diagonal:

            if self.is_true_eye( location_neighbor, future_eyes ):

                count_bad_diagonal -= 1

        if count_bad_diagonal <= max_bad_diagonal:

            eyes.update( future_eyes )
            return 1

        # not an eye
        return 0



    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef list get_sensible_moves( self ):

        cdef list  sensible_moves = []
        cdef dict  eyes           = {}

        for location_legal in self.legalMoves:

            if not self.is_true_eye( location_legal, eyes ):

                sensible_moves.append( location_legal )

        return sensible_moves

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char is_sensible( self, location ):

        cdef int  i
        cdef short neighbor
        cdef char max_bad_diagonal = 1
        cdef char count_diagonal   = 0
        cdef char count_border     = 0

        # loop over neighbor if EMPTY -> a sensible move
        # loop over all neighbor
        for i in range( 4 ):
            
            neighbor   = self.neighbor[ location * 4 + i ]
            boardValue = self.board[ neighbor ]
            if boardValue == EMPTY:

                # empty neighbor indicates it is not an eye
                return 1
            if boardValue == BORDER:

                count_border += 1

        # loop over diagonals, if eye -> not a sensible move
        return 0

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short get_liberties_after( self, short location, short max ):
        """
           calculate group liberty after a move location
        """

        cdef char  liberty
        cdef Group group

        group = self.get_group_after( location )

        liberty = len( group.location_liberty ) - 1
        if liberty > max:
            return max

        return liberty


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short get_capture_size( self, short location, short max ):
        #

        if location == self.ko:
            return 0

        cdef int   i
        cdef short size
        cdef short neighbor
        cdef Group group_temp
        cdef dict  stones   = {}

        # loop over all neighbor
        for i in range( 4 ):
            
            neighbor = self.neighbor[ location * 4 + i ]

            # if neighbor is opponent
            if self.board[ neighbor ] == self.player_opponent:

                group_temp = self.boardGroups[ neighbor ]
                # if group has only one liberty
                if len( group_temp.location_liberty ) == 1:

                    stones.update(  group_temp.location_stones )

        size = len( stones )

        if size > max:
            return max

        return size

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short get_self_atari_size( self, short location, short max ):
        #

        return 0

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    # TODO
    # play move and update liberties hashes etc.
    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void add_move( self, short location ):

        # assume legal move!
        cdef short i

        self.ko = PASS

        self.board[ location ] = self.player_current
        self.add_to_group( location )

        # change player colour
        self.player_current = self.player_opponent
        self.player_opponent = ( BLACK if self.player_current == WHITE else WHITE )

        # add to history
        self.history.append( location )

        # generate legal moves? -> or should be done when board changes
        self.legalMoves = []
        for i in range( self.size * self.size ):

            if self.is_legal_move( i ):

                self.legalMoves.append( i )
        # update zobrist

    # copy gamestate (self) and play move
    @cython.boundscheck( False )
    @cython.wraparound(  False )
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

        cdef int   x, y
        cdef short location
        ( x, y ) = action
        location = self.calculate_board_location( y, x )
        self.add_move( location )

        if location in self.legalMoves:
            
            return True

        return False

    def get_player_active_colour( self ):
        return self.player_current


    # copy state and play move
    # action has to be a ( x, y ) tuple
    # this function should be used from Python environment, 
    # use new_state_add_move from C environment for speed
    def get_next_state( self, action ):

        # calculate location from tuple action
        cdef char x, y
        ( x, y ) = action

        return self.new_state_add_move( self.calculate_board_location( y, x ) )

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

        if include_eyes:
            return self.legalMoves

        return self.legalMoves

    # return true/false if move at action is legal
    def is_legal( self, action ):

        cdef char x, y
        cdef short location
        ( x, y ) = action

        # check outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # calculate location
        location = self.calculate_board_location( y, x )
        if location in self.legalMoves:
            return True

        return False

    # copy this state
    def copy( self ):

        return GameState( copyState = self )

    ############################################################################
    #   tests                                                                  #
    #                                                                          #
    ############################################################################

    def printer( self ):
        for i in range( self.size ):
            A = str( i ) + " "
            for j in range( self.size ):

                B = 0
                if self.board[ j + i * self.size ] == BLACK:
                    B = 2
                elif self.board[ j + i * self.size ] == WHITE:
                    B = 1
                A += str( B ) + " "
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

    def test_game_speed( self, list moves ):

        cdef short location

        for location in moves:
            self.add_move( location )

    def convert_moves( self, list moves ):
        cdef list converted_moves = []
        cdef int   x, y
        cdef short location

        for loc in moves:

            ( x, y ) = loc
            location = self.calculate_board_location( y, x )
            converted_moves.append( location )

        return converted_moves


