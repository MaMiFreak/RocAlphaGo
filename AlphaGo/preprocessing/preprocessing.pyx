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


    def __init__(self, feature_list, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False):
        """create a preprocessor object that will concatenate together the
        given list of features
        """
        
        # named features and their sizes are defined here
        FEATURES = {
            "board": {
                "size": 3,
                "function": self.get_board
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
