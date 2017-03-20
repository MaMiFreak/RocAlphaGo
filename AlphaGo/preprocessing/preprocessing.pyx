import ast
import numpy as np
import pyximport; pyximport.install()
import AlphaGo.go as go
import keras.backend as K

# This file is used anywhere that neural net features are used; setting the keras dimension ordering
# here makes it universal to the project.
K.set_image_dim_ordering('th')

##
# individual feature functions (state --> tensor) begin here
##

DEFAULT_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size",
    "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
    "sensibleness", "zeros"]

DEFAULT_ROLLOUT_FEATURES = [
    "response", "save_atari", "neighbor", "nakade", "response_12d",
    "non_response_3x3"]


class Preprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def get_board(self, state, tensor, offSet):
        """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
        always refers to the current player and plane 1 to the opponent
        """

        # own stone
        colour = state.current_player
        tensor[offSet, :, :] = state.board == colour

        # opponent stone
        colour = -state.current_player
        tensor[offSet + 1, :, :] = state.board == colour

        # empty space
        tensor[offSet + 2, :, :] = state.board == go.EMPTY

        return offSet + 3


    def get_turns_since(self, state, tensor, offSet, maximum=8):
        """A feature encoding the age of the stone at each location up to 'maximum'

        Note:
        - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
        - EMPTY locations are all-zero features
        """

        for x in range(state.size):
            for y in range(state.size):
                if state.stone_ages[x][y] >= 0:
                    tensor[min(state.stone_ages[x][y], offSet + maximum - 1), x, y] = 1

        return offSet + maximum


    def get_liberties(self, state, tensor, offSet, maximum=8):
        """A feature encoding the number of liberties of the group connected to the stone at
        each location

        Note:
        - there is no zero-liberties plane; the 0th plane indicates groups in atari
        - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
        - EMPTY locations are all-zero features
        """

        for i in range(maximum):
            # single liberties in plane zero (groups won't have zero), double
            # liberties in plane one, etc
            tensor[offSet + i, state.liberty_counts == i + 1] = 1
        # the "maximum-or-more" case on the backmost plane
        tensor[offSet + maximum - 1, state.liberty_counts >= maximum] = 1

        return offSet + maximum


    def get_capture_size(self, state, tensor, offSet, maximum=8):
        """A feature encoding the number of opponent stones that would be captured by
        playing at each location, up to 'maximum'

        Note:
        - we currently *do* treat the 0th plane as "capturing zero stones"
        - the [maximum-1] plane is used for any capturable group of size
          greater than or equal to maximum-1
        - the 0th plane is used for legal moves that would not result in capture
        - illegal move locations are all-zero features

        """

        for (x, y) in state.get_legal_moves():
            # multiple disconnected groups may be captured. hence we loop over
            # groups and count sizes if captured.
            n_captured = 0
            for neighbor_group in state.get_groups_around((x, y)):
                # if the neighboring group is opponent stones and they have
                # one liberty, it must be (x,y) and we are capturing them
                # (note suicide and ko are not an issue because they are not
                # legal moves)
                (gx, gy) = next(iter(neighbor_group))
                if (state.liberty_counts[gx][gy] == 1) and \
                   (state.board[gx, gy] != state.current_player):
                    n_captured += len(state.group_sets[gx][gy])
            tensor[offSet + min(n_captured, maximum - 1), x, y] = 1

        return offSet + maximum


    def get_self_atari_size(self, state, tensor, offSet, maximum=8):
        """A feature encoding the size of the own-stone group that is put into atari by
        playing at a location

        """

        for (x, y) in state.get_legal_moves():
            # make a copy of the liberty/group sets at (x,y) so we can manipulate them
            lib_set_after = set(state.liberty_sets[x][y])
            group_set_after = set()
            group_set_after.add((x, y))
            captured_stones = set()
            for neighbor_group in state.get_groups_around((x, y)):
                # if the neighboring group is of the same color as the current player
                # then playing here will connect this stone to that group
                (gx, gy) = next(iter(neighbor_group))
                if state.board[gx, gy] == state.current_player:
                    lib_set_after |= state.liberty_sets[gx][gy]
                    group_set_after |= state.group_sets[gx][gy]
                # if instead neighboring group is opponent *and about to be captured*
                # then we might gain new liberties
                elif state.liberty_counts[gx][gy] == 1:
                    captured_stones |= state.group_sets[gx][gy]
            # add captured stones to liberties if they are neighboring the 'group_set_after'
            # i.e. if they will become liberties once capture is resolved
            if len(captured_stones) > 0:
                for (gx, gy) in group_set_after:
                    # intersection of group's neighbors and captured stones will become liberties
                    lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
            if (x, y) in lib_set_after:
                lib_set_after.remove((x, y))
            # check if this move resulted in atari
            if len(lib_set_after) == 1:
                group_size = len(group_set_after)
                # 0th plane used for size=1, so group_size-1 is the index
                tensor[offSet + min(group_size - 1, maximum - 1), x, y] = 1

        return offSet + maximum


    def get_liberties_after(self, state, tensor, offSet, maximum=8):
        """A feature encoding what the number of liberties *would be* of the group connected to
        the stone *if* played at a location

        Note:
        - there is no zero-liberties plane; the 0th plane indicates groups in atari
        - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
        - illegal move locations are all-zero features
        """

        # note - left as all zeros if not a legal move
        for (x, y) in state.get_legal_moves():
            # make a copy of the set of liberties at (x,y) so we can add to it
            lib_set_after = set(state.liberty_sets[x][y])
            group_set_after = set()
            group_set_after.add((x, y))
            captured_stones = set()
            for neighbor_group in state.get_groups_around((x, y)):
                # if the neighboring group is of the same color as the current player
                # then playing here will connect this stone to that group and
                # therefore add in all that group's liberties
                (gx, gy) = next(iter(neighbor_group))
                if state.board[gx, gy] == state.current_player:
                    lib_set_after |= state.liberty_sets[gx][gy]
                    group_set_after |= state.group_sets[gx][gy]
                # if instead neighboring group is opponent *and about to be captured*
                # then we might gain new liberties
                elif state.liberty_counts[gx][gy] == 1:
                    captured_stones |= state.group_sets[gx][gy]
            # add captured stones to liberties if they are neighboring the 'group_set_after'
            # i.e. if they will become liberties once capture is resolved
            if len(captured_stones) > 0:
                for (gx, gy) in group_set_after:
                    # intersection of group's neighbors and captured stones will become liberties
                    lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
            # (x,y) itself may have made its way back in, but shouldn't count
            # since it's clearly not a liberty after playing there
            if (x, y) in lib_set_after:
                lib_set_after.remove((x, y))
            tensor[offSet + min(maximum - 1, len(lib_set_after) - 1), x, y] = 1

        return offSet + maximum


    def get_ladder_capture(self, state, tensor, offSet):
        """A feature wrapping GameState.is_ladder_capture().
        """

        for (x, y) in state.get_legal_moves():
            tensor[offSet, x, y] = state.is_ladder_capture((x, y))

        return offSet + 1


    def get_ladder_escape(self, state, tensor, offSet):
        """A feature wrapping GameState.is_ladder_escape().
        """

        for (x, y) in state.get_legal_moves():
            tensor[offSet, x, y] = state.is_ladder_escape((x, y))

        return offSet + 1


    def get_sensibleness(self, state, tensor, offSet):
        """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
        """

        for (x, y) in state.get_legal_moves(include_eyes=False):
            tensor[offSet, x, y] = 1

        return offSet + 1


    def get_legal(self, state, tensor, offSet):
        """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
        """

        for (x, y) in state.get_legal_moves():
            tensor[offSet, x, y] = 1

        return offSet + 1


    def get_response(self, state, tensor, offSet):
        """Fast rollout feature
        """

        if len(state.history) > 0:
            for (x, y) in state.get_legal_moves(include_eyes=False):
                index = state.get_pattern_response_12d((x, y))
                if index >= 0:
                    tensor[offSet, x, y] = 1

        return offSet + 1


    def get_save_atari(self, state, tensor, offSet):
        """Fast rollout feature
        """

        return self.get_self_atari_size(state, tensor, offSet, maximum=1)


    def get_neighbor(self, state, tensor, offSet):
        """Fast rollout feature
        """

        if len(state.history) > 0:
            for (x, y) in state.get_legal_moves(include_eyes=False):
                index = state.get_8_connected((x, y))
                if index >= 0:
                    tensor[offSet + index, x, y] = 1

        return offSet + 2


    def get_nakade(self, state, tensor, offSet):
        """Fast rollout feature
        """

        for (x, y) in state.get_legal_moves(include_eyes=False):
            index = state.get_pattern_nakade((x, y))
            if index >= 0:
                tensor[offSet + index, x, y] = 1

        return offSet + self.pattern_nakade_size


    def get_response_12d(self, state, tensor, offSet):
        """Fast rollout feature
        """

        if len(state.history) > 0:
            for (x, y) in state.get_legal_moves(include_eyes=False):
                index = self.pattern_response_12d.get(state.get_pattern_response_12d((x, y)), -1)
                if index >= 0:
                    tensor[offSet + index, x, y] = 1

        return offSet + self.pattern_response_12d_size


    def get_non_response_3x3(self, state, tensor, offSet):
        """Fast rollout feature
        """

        for (x, y) in state.get_legal_moves(include_eyes=False):
            index = self.pattern_non_response_3x3.get(state.get_pattern_non_response_3x3((x, y)), -1)
            if index >= 0:
                tensor[offSet + index, x, y] = 1

        return offSet + self.pattern_non_response_3x3_size

    def zeros(self, state, tensor, offSet):
        """Fast rollout feature
        """
        
        # do nothing, numpy array has to be initialized with zero

        return offSet + 1

    def ones(self, state, tensor, offSet):
        """Fast rollout feature
        """

        tensor[offSet, :, :] = 1
        
        return offSet + 1

    def colour(self, state, tensor, offSet):
        """Fast rollout feature
        """

        if state.current_player == go.BLACK:
            tensor[offSet, :, :] = 1
        
        return offSet + 1

    def __init__(self, feature_list, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False):
        """create a preprocessor object that will concatenate together the
        given list of features
        """
        
        # 3x3 patterns/dict_non_response_3x3_min_10_occurrence.pat
        # 12d patterns/dict_response_12d_min_10_occurrence.pat
        
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

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """

        #x = state.test()
        #print( x )

        # create complete array now instead of concatenate later
        tensor = np.zeros((self.output_dim, state.size, state.size))
        offSet = 0

        for proc in self.processors:
            offSet = proc(state, tensor, offSet)

        # create a singleton 'batch' dimension
        f, s = self.output_dim, state.size
        
        return tensor.reshape((1, f, s, s))
