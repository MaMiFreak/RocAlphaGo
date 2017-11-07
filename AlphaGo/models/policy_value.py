from keras.models import Model
from keras.layers import *
from AlphaGo import go
from keras.layers.merge import add
from keras.layers.core import Activation, Flatten
from AlphaGo.util import flatten_idx
from AlphaGo.models.nn_util import Bias, NeuralNetBase, neuralnet
from AlphaGo.preprocessing.preprocessing_rollout import Preprocess
import numpy as np


@neuralnet
class CNNPolicyValue(NeuralNetBase):
    """
       uses a convolutional neural network with a residual block part to evaluate the state of a game
       computes probability distribution over the next action and the win probability of the current player
    """

    def get_preprocessor(self, feature_list, board_size):
        """
           return preprocessor (override as we use rollout preprocessor)
        """

        return Preprocess(feature_list, size=board_size)

    def _model_forward(self):
        """
           we have to override this functions because this network has multiple outputs
           Construct a function using the current keras backend that, when given a batch
           of inputs, simply processes them forward and returns the output
           This is as opposed to model.compile(), which takes a loss function
           and training method.
           c.f. https://github.com/fchollet/keras/issues/1426
        """
        # The uses_learning_phase property is True if the model contains layers that behave
        # differently during training and testing, e.g. Dropout or BatchNormalization.
        # In these cases, K.learning_phase() is a reference to a backend variable that should
        # be set to 0 when using the network in prediction mode and is automatically set to 1
        # during training.

        if self.model.uses_learning_phase:
            forward_function = K.function([self.model.input, K.learning_phase()],
                                          self.model.output)

            # the forward_function returns a list of tensors
            # the first [0] gets the front tensor.
            return lambda inpt: forward_function([inpt, 0])
        else:
            # identical but without a second input argument for the learning phase
            forward_function = K.function([self.model.input], self.model.output)
            return lambda inpt: forward_function([inpt])

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """
           helper function to normalize a distribution over the given list of moves
           and return a list of (move, prob) tuples
        """

        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # add pass move location
        move_indices.append( size * size )
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        # add pass move value
        moves.append( go.PASS )

        return zip(moves, distribution)

    def eval_state(self, state, moves=None):
        """
           Given a GameState object, returns a tuple with alist of (action, probability) pairs
           according to the network outputs and win probability of current player
           If a list of moves is specified, only those moves are kept in the distribution
        """

        tensor = self.preprocessor.state_to_tensor(state)

        # run the tensor through the network
        network_output = self.forward(tensor)
        moves = moves or state.get_legal_moves()

        actions = self._select_moves_and_normalize(network_output[0][0], moves, state.get_size())

        return ( actions, network_output[1][0][0])

    @staticmethod
    def create_network(**kwargs):
        """
           construct a alphago zero style residual neural network.
           Keword Arguments:
           - board:            width of the go board to be processed                      (default 19)
           - input_dim:        depth of features to be processed by first layer           (no default)
           - activation:       type of activation used eg relu sigmoid tanh               (default relu)
             pre residual block convolution
           - conv_kernel:      kernel size used in first convolution layer                (default 3)   (Must be odd)
             residual block 
           - residual_depth:   number of residual blocks                                  (default 39)
           - residual_filter:  number of filters used on residual block convolution layer (default 256)
                               also used for pre residual block convolution as 
                               they have to be equal
           - residual_kernel:  kernel size used in first residual block convolution layer (default 3)   (Must be odd)
             value head
           - value_size:       size of fully connected layer for value output             (default 256)
           - value_outputs:    amount of value outsputs                                   (default 1)   for training with multiple komi values https://arxiv.org/pdf/1705.10701.pdf
           - value_activation: value head output activation eg relu sigmoid tanh          (default tanh)
        """

        defaults = {
            "board": 19,
            "activation": 'relu',
            "conv_kernel" : 3,
            "residual_depth" : 39,
            "residual_filter" : 256,
            "residual_kernel" : 3,
            "value_size": 256,
            "value_activation": 'tanh',
            "value_outputs": 1
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create input with theano ordering ( "channels_first" )
        inp = Input( shape=( params["input_dim"], params["board"], params["board"] ) )

        # create convolution layer
        layer = Conv2D( params["residual_filter"], ( params["conv_kernel"], params["conv_kernel"] ), data_format="channels_first", padding='same', name='conv1' )( inp )
        layer = BatchNormalization( name='conv1_bn')( layer )
        layer = Activation( params["activation"] )( layer )

        # create residual blocks
        for i in range( params["residual_depth"] ):

            # residual block comon name
            name = 'residual_block_' + str( i ) + '_'

            # first residual block convolution
            residual = Conv2D( params["residual_filter"], ( params["residual_kernel"], params["residual_kernel"] ), data_format="channels_first", padding='same', name=name + 'conv1' )( layer )
            residual = BatchNormalization( name=name + 'conv1_bn')( residual )
            residual = Activation( params["activation"] )( residual )

            # second residual block convolution
            residual = Conv2D( params["residual_filter"], ( params["residual_kernel"], params["residual_kernel"] ), data_format="channels_first", padding='same', name=name + 'conv2' )( residual )
            residual = BatchNormalization( name=name + 'conv2_bn')( residual )
            residual = Activation( params["activation"] )( residual )

            # add residual block input
            layer = add( [ layer, residual ] )
            layer = Activation( params["activation"] )( layer )


        # create policy head
        policy = Conv2D( 2, ( 1, 1 ) )( layer )
        policy = BatchNormalization()( policy )
        policy = Activation( params["activation"] )( policy )
        policy = Flatten()( policy )
        # board * board for board locations, +1 for pass move
        policy = Dense( ( params["board"] * params["board"] ) + 1, activation='softmax', name='policy_output' )( policy )

        # create value head
        value = Conv2D(1, (1, 1) )( layer )
        value = BatchNormalization()( value )
        value = Activation( params["activation"] )( value )
        value = Flatten()( value )
        value = Dense( params["value_size"], activation=params["activation"] )( value )
        value = Dense( params["value_outputs"], activation=params["value_activation"], name='value_output' )( value )

        # create the network:
        network = Model( inp, [ policy, value ] )

        return network
