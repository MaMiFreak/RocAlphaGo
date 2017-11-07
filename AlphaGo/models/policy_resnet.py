from keras.models import Model
from keras.layers import *
from keras.layers.merge import add
from keras.layers.core import Activation, Flatten
from AlphaGo.util import flatten_idx
from AlphaGo.models.nn_util import Bias, NeuralNetBase, neuralnet
import numpy as np


@neuralnet
class CNNPolicyResnet(NeuralNetBase):
    """
       uses a convolutional neural network with residual blocks to evaluate the state of the game
       and compute a probability distribution over the next action
    """

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)

    def batch_eval_state(self, states, moves_lists=None):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.

        Analogous to [eval_state(s) for s in states]

        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []
        state_size = states[0].get_size()
        if not all([st.get_size() == state_size for st in states]):
            raise ValueError("all states must have the same size")
        # concatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)
        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        network_output = self.forward(nn_input)
        # default move lists to all legal moves
        moves_lists = moves_lists or [st.get_legal_moves() for st in states]
        results = [None] * n_states
        for i in range(n_states):
            results[i] = self._select_moves_and_normalize(network_output[i], moves_lists[i],
                                                          state_size)
        return results

    def eval_state(self, state, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.forward(tensor)
        moves = moves or state.get_legal_moves()
        return self._select_moves_and_normalize(network_output[0], moves, state.get_size())

    @staticmethod
    def create_network(**kwargs):
        """
           construct a residual neural policy network.

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
        """

        defaults = {
            "board": 19,
            "activation": 'relu',
            "conv_kernel" : 3,
            "residual_depth" : 39,
            "residual_filter" : 256,
            "residual_kernel" : 3
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
        policy = Conv2D( 1, ( 1, 1 ) )( layer )
        policy = BatchNormalization()( policy )
        policy = Activation( params["activation"] )( policy )
        policy = Flatten()( policy )
        # board * board for board locations
        policy = Dense( params["board"] * params["board"], activation='softmax', name='policy_output' )( policy )

        # create the network:
        network = Model( inp, policy )


        return network

