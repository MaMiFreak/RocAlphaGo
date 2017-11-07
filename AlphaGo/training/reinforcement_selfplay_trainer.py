import os
import time
import json
import h5py
import math
import argparse
import threading
import numpy as np
from AlphaGo.go import GameState, BLACK
from AlphaGo.util import save_gamestate_to_sgf
from multiprocessing import Process, Queue, Value
from AlphaGo.preprocessing.preprocessing_rollout import Preprocess as PreprocessRollout

import time

PUCT_CONSTANT            = 0.01 # MCTS PUCT constant
GREEDY_START             = 4    # MCTS greedy start move
PRIOR_EVEN               = 4

DEFAULT_KOMI             = 5.5
DEFAULT_OPTIMIZER        = 'SGD'
DEFAULT_LEARNING_RATE    = .003
DEFAULT_BATCH_SIZE       = 400    # combine multiple batches?? -> 2048
DEFAULT_EPOCH_SIZE       = 10000  # 2048000
DEFAULT_TEST_AMOUNT      = 400
DEFAULT_TRAIN_EVERY      = 5
DEFAULT_SIMULATIONS      = 2    # 1600
DEFAULT_RESIGN_TRESHOLD  = 0.05   # should be automatically adjusted??
DEFAULT_ALLOW_RESIGN     = 0.8

# in order to optimize gpu utilization each prediction worker has a unique group of
# game generator workers. worker ratio is used to make sure that there are always samples 
# for the prediction worker to process
# EXAMPLE
# - DEFAULT_PREDICTION_WORK  = 4
# - DEFAULT_PREDICTION_BATCH = 2
# - DEFAULT_WORKER_RATIO     = 3
# means there are 4 prediction workers, running the network with batches of 2
# and each prediction worker has 6 game generator workers
# in total there are 24 game generator workers
# depending on gpu and cpu optimal settings might vary
DEFAULT_PREDICTION_WORK  = 1
DEFAULT_PREDICTION_BATCH = 2
DEFAULT_WORKER_RATIO     = 3

# metdata file
FILE_METADATA = 'metadata_policy_value_reinforcement.json'
# hdf5 training file
FILE_HDF5 = 'training_samples.hdf5'
# weight folder
FOLDER_WEIGHT = os.path.join('policy_value_reinforcement_weights')
# folder sgf files
FOLDER_SGF = os.path.join('policy_value_reinforcement_sgf')

# TODO use axis to transform 3d array correctly (numpy 13.3+)
BOARD_TRANSFORMATIONS = {
    0: lambda feature: feature,
    1: lambda feature: np.rot90(feature, 1),
    2: lambda feature: np.rot90(feature, 2),
    3: lambda feature: np.rot90(feature, 3),
    4: lambda feature: np.fliplr(feature),
    5: lambda feature: np.flipud(feature),
    6: lambda feature: np.transpose(feature),
    7: lambda feature: np.fliplr(np.rot90(feature, 1))
}

BOARD_BACK_TRANSFORMATIONS = {
    0: lambda feature: feature,
    1: lambda feature: np.rot90(feature, 3),
    2: lambda feature: np.rot90(feature, 2),
    3: lambda feature: np.rot90(feature, 1),
    4: lambda feature: np.fliplr(feature),
    5: lambda feature: np.flipud(feature),
    6: lambda feature: np.transpose(feature),
    7: lambda feature: np.rot90(np.fliplr(feature), 3)
}

class Counter(object): 
    """
       Simple multiprocessing counter to keep track of game count
    """
   
    def __init__( self, value=0 ):

        self.count = Value( 'i', value )

    def increment(self):
        """
           increment count value and return value
        """
        with self.count.get_lock():

            self.count.value += 1
            return self.count.value

    def get_value(self):
        """
           get count value
        """
        with self.count.get_lock():

            return self.count.value

class HDF5_handler():
    """
       Simple Hdf5 file handler
    """

    def __init__( self, database ):

        self.idx           = len( database['action_value'] )
        # empty database has lenght 1
        if self.idx == 1:
            self.idx = 0

        self.database      = database
        self.states        = database['states']
        self.value         = database['action_value']
        self.policy        = database['action_policy']

    @staticmethod
    def create_hdf5_file( file_location, feature_list, board_size, depth ):
        """
           create hdf5 file
        """

        # todo check compression speed options

        # create hdf5 file
        database = h5py.File(file_location, 'w') # w Create file, truncate if exists

        try:

            # create states ( network inputs )
            database.create_dataset(
                                     'states',
                                     dtype=np.uint8,
                                     shape=(1, depth, board_size, board_size),
                                     maxshape=(None, depth, board_size, board_size), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, depth, board_size, board_size),
                                     compression="lzf"
                                    )

            # create action_value ( network value output ) -> do we want to train with multiple komi values? https://arxiv.org/pdf/1705.10701.pdf
            database.create_dataset(
                                     'action_value',
                                     dtype=np.int8,
                                     shape=(1, 1),
                                     maxshape=(None, 1), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, 1),
                                     compression="lzf"
                                    )

            # create action_policy ( network policy output )
            database.create_dataset(
                                     'action_policy',
                                     dtype=np.float16,
                                     shape=(1, ( board_size * board_size ) + 1 ),       # +1 for pass move
                                     maxshape=(None, ( board_size * board_size ) + 1 ), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, ( board_size * board_size ) + 1 ),
                                     compression="lzf"
                                    )

            # add features
            database['features'] = np.string_(','.join(feature_list))


        except Exception as e:
            raise e

        # close file
        database.close()

    def add_samples( self, state_samples, value_samples, policy_samples ):
        """
           add samples to hdf5 file
        """

        # calculate new database size
        size_new = self.idx + len( value_samples )

        try:

            # resize databases
            self.value.resize(  size_new, axis=0 )
            self.states.resize( size_new, axis=0 )
            self.policy.resize( size_new, axis=0 )

            # add samples
            self.value[self.idx:]  = value_samples
            self.states[self.idx:] = state_samples
            self.policy[self.idx:] = policy_samples

            self.idx = size_new

        except Exception as e:
            raise e

# train_model( id, model_file, weight_file, save_location, save_file, version, metadata_file, hdf5_file )
class train_model( Process ):

    def __init__( self, id, model_file, weight_file, save_location, save_file, version, metadata_file, hdf5_file ):
        """
        """
        Process.__init__(self)

        self.id             = id
        self.version        = version
        self.hdf5_file      = hdf5_file
        self.save_file      = save_file
        self.model_file     = model_file
        self.weight_file    = weight_file
        self.save_location  = save_location
        self.metadata_file  = metadata_file

    def run(self):
        """
        """

        from keras.optimizers import SGD
        from AlphaGo.models.policy_value import CNNPolicyValue
        network = CNNPolicyValue.load_model( self.model_file )

        # load metadata
        with open( self.metadata_file, "r" ) as f:
            metadata = json.load(f)

        result = {}

        if self.weight_file is not None:

            # load model
            network.model.load_weights( self.weight_file )

            # create batch generator
            # with open hdf5 file
            with h5py.File( self.hdf5_file, 'r' ) as data:

                generator = batch_generator( data['states'], data['action_policy'], data['action_value'], metadata['range_from_end'], metadata['batch_size'] )

                # create optimizer
                optimizer = SGD( lr=0.001 )
                network.model.compile( loss=['categorical_crossentropy','mse'], optimizer=optimizer, metrics=["accuracy"] )

                # train model for one epoch
                history = network.model.fit_generator(
                                        # verbose=0,
                                        generator=generator,
                                        steps_per_epoch=metadata['epoch_size'],
                                        epochs=1
                                       )

            # results
            result = {
                     'accuracy_value': history.history['value_output_acc'][0],
                     'accuracy_policy': history.history['policy_output_acc'][0],
                     'loss':history.history['loss'][0],
                     'loss_value':history.history['value_output_loss'][0],
                     'loss_policy':history.history['policy_output_loss'][0]
                     }
        else:

            # get model board size
            metadata["board_size"] = network.model.input_shape[-1]
            # get model feature list
            metadata["feature_list"] = network.preprocessor.get_feature_list()
            # get model feature list
            metadata["feature_depth"] = network.preprocessor.get_output_dimension()

        result[ 'version' ] = self.version
        result[ 'file'    ] = self.save_file

        # add new model training results
        metadata['model_verions'].append( result )

        # save metadata     
        with open( self.metadata_file, "w" ) as f:
            json.dump( metadata, f, indent=2)

        # save model
        network.model.save( self.save_location )


class compare_newest_vs_best_models( Process ):

    def __init__(self, id, model_file, metadata_file ):
        """
        """
        Process.__init__(self)


        self.id                   = id
        self.model_file           = model_file
        self.metadata_file        = metadata_file
        self.player_weight_file   = player_weight_file
        self.opponent_weight_file = opponent_weight_file

    def run(self):
        """
        """

        # load metadata
        with open( self.metadata_file, "r" ) as f:
            metadata = json.load(f)

        # compare strength, twogtp?

        # get win ratio
        ratio = 0.6

        # get training result
        result = metadata['model_verions'][-1]
        # add strength comparison result
        result['opponent'] = self.opponent_weight_file
        result['winratio'] = ratio
        result['comparegamecount'] = metadata['test_amount']
        result['currentgamecount'] = metadata['game_count']

        # check if new model beats previous model with margin
        if ratio >= 0.55:

            metadata['best_model'] = result['file']

        # save metadata     
        with open( self.metadata_file, "w" ) as f:
            json.dump( metadata, f, indent=2)

# id, batch_size, model_file, weight_file, queue_requests, queue_predictions, signal
class run_predictions( Process ):

    def __init__(self, id, batch_size, model_file, weight_file, queue_requests, queue_predictions, signal ):
        """
        """
        Process.__init__(self)


        self.id                = id
        self.signal            = signal
        self.model_file        = model_file
        self.batch_size        = batch_size
        self.weight_file       = weight_file
        self.queue_requests    = queue_requests
        self.queue_predictions = queue_predictions

    def run(self):
        """
        uncomment this part in order to set tensorflow memory usage
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        set_session( tf.Session( config=config ) )
        """

        from AlphaGo.models.policy_value import CNNPolicyValue
        self.network = CNNPolicyValue.load_model( self.model_file )
        self.network.model.load_weights( self.weight_file )

        while self.signal.get_value() <= 0:

            requests   = []
            worker_ids = []

            # get #batch_size request
            while len( requests ) < self.batch_size:

                worker_id, request = self.queue_requests.get()
                requests.append( request )
                worker_ids.append( worker_id )

            # predict all requests
            predictions = self.network.forward( requests )

            # return prediction to corresponding worker
            for worker_id, policy, value in zip( worker_ids, predictions[ 0 ], predictions[ 1 ]  ):

                # add policy and value prediction to corresponding worker queue
                self.queue_predictions[ worker_id ].put( ( policy, value ) )
        

class batch_generator:
    """
       A generator of batches of training data for use with the fit_generator function
       of Keras.

       it is threading safe but not multiprocessing therefore only use it with
       pickle_safe=False when using multiple workers
    """

    def __init__(self, state_dataset, policy_dataset, value_dataset, range_from_end, batch_size ):
        """

        """

        self.batch_size     = batch_size
        self.value_dataset  = value_dataset
        self.state_dataset  = state_dataset
        self.policy_dataset = policy_dataset
        self.range_from_end = range_from_end
        self.data_lock      = threading.Lock()
        self.max            = len( state_dataset )
        self.min            = max( 0, self.max - range_from_end )

    def __iter__(self):
        return self

    def next_indice(self):
        # use lock to prevent double hdf5 acces
        with self.data_lock:

            # get random training sample id
            sample_id = np.random.randint( low=self.min, high=self.max )

            # get state sample
            state = self.state_dataset[sample_id]
            # get value sample
            value = self.value_dataset[sample_id]
            # get policy sample
            policy = self.policy_dataset[sample_id]

            # return state, policy and value
            return state, policy, value

    def next(self):

        inputs        = []
        output_value  = []
        output_policy = []

        for batch_idx in xrange(self.batch_size):
            state, policy, value = self.next_indice()

            transformation = np.random.randint( low=0, high=8 )
            if transformation > 0: # 0 rotation does nothing

                transform = BOARD_TRANSFORMATIONS[ transformation ]

                state  = np.array( [ transform( plane ) for plane in state ] )
                # TODO verify correct rotation
                data = transform( policy[ 0 : -1 ].reshape( ( 9, 9 ) ) )
                policy = np.append( data, policy[ -1 ] )

            inputs.append( state )
            output_value.append( value )
            output_policy.append( policy )

        return ( np.array( inputs ), [np.array(output_policy), np.array(output_value)] )


class MCTS_Tree_Node():
    """
       Alphago zero style mcts node
    """

    def __init__( self, state, policy, value, nn_input, greedy_start, depth, parent=None ):
        """
           init new mcts tree node:
           - state is a GameState
           - policy is neural network policy prediction
           - value  is neural network value  prediction
           - nn_input is neural network input ( prevent double preprocessing )
           - parent is parent tree node
        """

        self.depth        = depth
        self.state        = state
        self.value        = value
        self.visits       = 1
        self.parent       = parent
        self.children     = {}
        self.nn_input     = nn_input
        self.greedy_start = greedy_start

        # create list with tuples: ( legal move location, probability )
        self.legal = []
        moves = state.get_legal_locations()
        # add pass move
        moves.append( 9 * 9 )

        for move in moves:

            self.legal.append( [ move, policy[ move ] ] )

    def select_expand_move( self ):
        """
           select tree expansion move
           return treenode and move
        """

        # check if it is a terminal state
        if self.state.is_end_of_game():

            self.update( self.score )
            return self, None

        best_move  = -1
        best_value = -100

        # select most urgent move
        for move, probability in self.legal:

            urgency = self.calculate_urgency( self.children.get( move, None ), probability )
            if urgency > best_value:

                best_value = urgency
                best_move  = move

        if best_move in self.children:

            return self.children[ best_move ].select_expand_move()

        else:

            return self, best_move


    def update( self, value ):
        """
           update node and all parent node
        """

        self.visits += 1
        self.value  += value

        if self.parent is not None:

            self.parent.update( 1 - value )


    def calculate_urgency( self, child, probability ):
        """
           PUCT urgency
        """

        if child is None:

            visits = 1
            expectation = 0.5
        else:

            visits = 1 + child.visits
            expectation = float(child.value + PRIOR_EVEN/2) / (child.visits + PRIOR_EVEN)

        return expectation + PUCT_CONSTANT * probability * math.sqrt( self.visits ) / visits

    def get_best_child( self ):
        """
           return best child node
        """

        children = self.children.items()

        if self.depth >= self.greedy_start:

            # select best move
            return max( children, key=lambda child: child[1].visits )[1]
        else:
            # select move 
            probability     = [ ( float( child[ 1 ].visits ) / self.visits ) ** 2 for child in children ]
            probability_sum = sum( probability )
            probability     = [ prop / probability_sum for prop in probability ]
            choice = np.random.choice( len( children ), p=probability )
            return children[ choice ][ 1 ]


    def apply_dirichlet_noise( self ):
        """
           apply dirichlet noise to legal move probabilities
           TODO find out if this is correct or should this be done in select expand move
        """

        # generate dirichlet noise
        dirichlet = np.random.dirichlet( ( 0.03, 1 ), len( self.legal ) )

        # apply dirichlet noise to proir probabilities
        for i in range( len( self.legal ) ):

            self.legal[ i ][ 1 ] = ( self.legal[ i ][ 1 ] * 0.75 ) +  ( dirichlet[ i ][ 0 ] * 0.25 )


class Games_Generator( Process ):
    """
       
    """

    def __init__( self, id, queue_save, queue_requests, queue_predictions, game_count, feature_list, board_size, allow_resign, resign_threshold, simulations, out_directory, seed, komi):

        Process.__init__(self)
        self.id                = id
        self.komi              = komi
        self.max_moves         = board_size * board_size * 2
        self.queue_save        = queue_save
        self.game_count        = game_count
        self.board_size        = board_size
        self.simulations       = simulations
        self.greedy_start      = GREEDY_START
        self.allow_resign      = allow_resign
        self.preprocessor      = PreprocessRollout( feature_list, size=board_size )
        self.out_directory     = out_directory
        self.queue_requests    = queue_requests
        self.resign_threshold  = resign_threshold
        self.queue_predictions = queue_predictions

        # make sure each worker uses different seed
        np.random.seed( seed )

    def run(self):

        sgf_id       = 0
        allow_resign = False

        while True:

            ################################################
            ############################### prepare new game

            # lists to keep track of training samples
            training_policy = []
            training_state  = []
            training_value  = [] 

            # new game
            state = GameState( size = self.board_size )
            # generate request
            request = self.preprocessor.state_to_single_tensor( state )
            # request network prediction
            self.queue_requests.put( ( self.id, request ) )

            # get network prediction ( blocks )
            policy, value = self.queue_predictions.get()
            value = ( value + 1 ) / 2
            # new MCTS tree
            root = MCTS_Tree_Node( state, policy, value, request, self.greedy_start, 0 )

            # allow resign in self.allow_resign % of all games
            if sgf_id > 100:

                allow_resign = np.random.random_sample() < self.allow_resign

            # used to alternate between black win/ white win
            # initialized as if black wins, white lose
            # training_value has to be multiplied with -1 when white wins
            # training_value has to be multiplied with  0 in case of a draw
            colour = 1

            ################################################
            ################################# start new game

            # play game until resign or termination
            while True:

                root.visits -= 1

                ############################################
                ####################### run mcts exploration

                # apply dirichlet noise to increase exploration
                root.apply_dirichlet_noise()

                for _ in range( self.simulations ):

                    node, move = root.select_expand_move()

                    # check if terminal state is reached
                    if move is not None:

                        # not a terminal state

                        # get new boardstate with move
                        new_state = node.state.copy_and_move_at_location( move )

                        # get nn prediction
                        request = self.preprocessor.state_to_single_tensor( new_state )

                        # TODO random rotation

                        self.queue_requests.put( ( self.id, request ) )
                        policy, value = self.queue_predictions.get()
                        value = ( value + 1 ) / 2

                        # TODO rotate back

                        # add new child
                        child = MCTS_Tree_Node( new_state, policy, value, request, self.greedy_start, node.depth + 1, parent=node )
                        node.children[ move ] = child

                        # check if new state is teminal
                        if new_state.is_end_of_game():

                            # determine winner
                            winner      = new_state.get_winner( komi = self.komi )
                            current     = new_state.get_current_player()
                            value       = 0 if winner == current else 1
                            child.value = value
                            child.score = value

                        # update all parent nodes
                        node.update( 1 - value )

                # check game terminal state ( do not create training sample )
                if root.state.is_end_of_game():

                    winner = root.state.get_winner( komi = self.komi )
                    score  = 1 if winner == BLACK else -1
                    break

                ############################################
                ##################### generate training data

                # training state sample is preprocessed state ( root.nn input )
                training_state.append( root.nn_input )

                # training policy sample is board_size + 1 array with 
                # child.visits / total.visits ( creating a heat map of mcts visits )
                policy_sample = np.zeros( ( self.board_size * self.board_size + 1 ), dtype=np.float16 )
                for child_location in root.children:

                    child_visits  = root.children[ child_location ].visits
                    policy_sample[ child_location ] = float( child_visits ) / float( root.visits )

                training_policy.append( policy_sample )

                # add -1 for white move, 1 for black move
                # after all games are over we can multiply with 1, -1 or 0 to get correct values
                training_value.append( [ colour ] )
                colour *= -1

                # check resign
                if allow_resign and float( root.value ) / root.visits < self.resign_threshold:

                    current = root.state.get_current_player()
                    score   = 1 if current == BLACK else -1
                    break

                # game lenght
                if root.depth > self.max_moves:

                    score = 0
                    break

                ############################################
                ########################### select best move

                # select best move and set new root
                root = root.get_best_child()
                root.parent = None

            ################################################
            ################################## game finished

            # update training_value with score value
            for sample in training_value:

                sample[ 0 ] *= score

            # add training_positions to queue_save
            # correct order should be: state, value, policy
            self.queue_save.put( [ training_state, training_value, training_policy ] )

            # increment game counter
            sgf_id = self.game_count.increment()

            # save game to sgf
            result = ( 'black win' if score == 1 else 'white win' ) + ' ' + str( score )
            file_name = "game.{version:08d}.sgf".format(version=sgf_id)
            save_file = os.path.join(self.out_directory, FOLDER_SGF)
            save_gamestate_to_sgf(root.state, save_file, file_name, result=result, size=self.board_size)


class Training_Samples_Saver( Process ):
    """
       save games in queue to hdf5
    """

    def __init__( self, id, queue_save, hdf5_file ):

        Process.__init__(self)
        self.id = id
        self.hdf5_file  = hdf5_file
        self.queue_save = queue_save

    def run(self):

        # open hdf5 file
        with h5py.File( self.hdf5_file, 'r+') as f:

            # create hdf5 handler
            database = HDF5_handler( f )

            while True:

                # correct order should be: state, value, policy
                # or None as stopping signal
                training_samples = self.queue_save.get() # blocking

                # check if process should stop
                if training_samples is None:

                    break

                # correct order should be: state, value, policy
                # store samples in hdf5 file
                database.add_samples( training_samples[0], training_samples[1], training_samples[2] )


def save_metadata( metadata ):
    """
       Save metadata
    """

    # update metadata file        
    with open( metadata['meta_file'], "w" ) as f:

        json.dump(metadata, f, indent=2)

def load_metadata( metadata_file ):
    """
       Save metadata
    """
    
    # load data from json file
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        raise ValueError("Metadata file not found!")

    return metadata


def compare_strenght( model_file, current_network_weight_file, new_network_weight_file, amount, komi ):
    """
       let both network play vs eachother for #amount games
       return winning ratio for new model 
    """

    # TODO
    # use twogtp??

    return 0.60


def run_training( metadata, out_directory, verbose ):
    """
       Run training pipeline
    """

    # metadata file location
    metadata_file = str( os.path.join(out_directory, FILE_METADATA) )
    metadata['meta_file'] = metadata_file

    # hdf5 file location
    hdf5_file = os.path.join(out_directory, FILE_HDF5)

    # queue with samples to save to hdf5
    queue_save = Queue()
    # game counter 
    game_count = Counter( value=metadata['game_count'] )
    # list with ( queue_requests, queue_predictions ) for each prediction worker
    groups     = []

    # create worker groups for each prediction worker
    for _ in range( DEFAULT_PREDICTION_WORK ):

        # queue with positions to network.forward
        queue_requests    = Queue()
        # list with queue for each worker
        queue_predictions = [ Queue() for _ in range( DEFAULT_PREDICTION_BATCH * DEFAULT_WORKER_RATIO )]
        groups.append( ( queue_requests, queue_predictions ) )

        # start game generator worker process
        for i in range( DEFAULT_PREDICTION_BATCH * DEFAULT_WORKER_RATIO ):

            # each worker needs unique seed or they will play the same moves
            seed = np.random.randint( 4294967295 )
            worker = Games_Generator( i, queue_save, queue_requests, queue_predictions[ i ], game_count, metadata["feature_list"],
                                      metadata["board_size"], metadata["allow_resign"], metadata["resign_treshold"], metadata["simulations"],
                                      out_directory, seed, metadata['komi'] )
            worker.start()

    # start generating games, train and compare model strength
    while True:

        ###############################################
        ################################ generate games
        # start hdf5 saver process
        saver = Training_Samples_Saver( 0, queue_save, hdf5_file )
        saver.start()

        if verbose:
            print 'Generating self play data'

        signal = Counter( value=0 )

        worker_prediction = []
        best_weight_file  = os.path.join(out_directory, FOLDER_WEIGHT, metadata['best_model'])

        # start prediction workers
        for queue_requests, queue_predictions in groups:

            # id, batch_size, model_file, weight_file, queue_requests, queue_predictions, signal
            worker = run_predictions( 0, DEFAULT_PREDICTION_BATCH, metadata["model_file"], best_weight_file, queue_requests, queue_predictions, signal )
            worker_prediction.append( worker )
            worker.start()
        
        # update game count every minute untill next training point
        while game_count.get_value() < metadata['next_training_point']:

            time.sleep( 60 ) 

            # update game count
            metadata['game_count'] = game_count.get_value()

            # update metadata file        
            save_metadata( metadata )

        # send stop signal to prediction workers
        signal.increment()
        # wait for all prediction workers to stop
        for worker in worker_prediction:

            worker.join()

        # stop hdf5 save thread just to be sure it will not interfere with training
        queue_save.put(None)
        # wait for process to stop
        saver.join()

        # update game count
        metadata['game_count'] = game_count.get_value()

        # save metadata        
        save_metadata( metadata )

        if verbose:
            print 'Training new model'

        ###############################################
        ################################### train model
        save_file = "weights.{version:05d}.hdf5".format(version=metadata['epoch_count'])
        save_location = str( os.path.join(out_directory, FOLDER_WEIGHT, save_file) )
        weight_location = str( os.path.join(out_directory, FOLDER_WEIGHT, metadata["newest_model"]) )
        # train_model( id, model_file, weight_file, save_location, save_file, version, metadata_file, hdf5_file )
        worker = train_model( 0, metadata["model_file"], weight_location, save_location, save_file, metadata['epoch_count'], metadata_file, hdf5_file )
        worker.start()
        worker.join()

        # load metadata ( been changed by trainer )
        metadata = load_metadata( metadata_file )

        metadata["newest_model"] = save_file

        if verbose:
            print 'Testing new model strength'

        ###############################################
        ############################## compare strength
        ratio = compare_strenght( metadata["model_file"], metadata['best_model'], metadata["newest_model"], metadata['test_amount'], metadata['komi'] )

        # get training result
        result = metadata['model_verions'][-1]
        # add strength comparison result
        result['opponent'] = metadata['best_model']
        result['winratio'] = ratio
        result['comparegamecount'] = metadata['test_amount']
        result['currentgamecount'] = metadata['game_count']

        # check if new model beats previous model with margin
        if ratio >= 0.55:

            metadata['best_model'] = result['file']

        # update for next training point
        metadata['epoch_count'] += 1
        metadata['next_training_point'] += metadata['train_every']

        # save metadata      
        save_metadata( metadata )


def start_training(args):
    """
       create metadata, check argument settings and start training
    """

    # create metadata
    metadata = {
        "next_training_point": args.train_every * 2, # first time we want to get enough games before training
        "resign_treshold": args.resign_treshold,
        "range_from_end": 100 * 50000,
        "learning_rate": args.learning_rate,
        "model_verions": [],
        "allow_resign": args.allow_resign,
        "train_every": args.train_every,
        "test_amount": args.test_amount,
        "simulations": args.simulations,
        "epoch_size": args.epoch_size,
        "batch_size": args.minibatch,
        "best_model": args.weights,
        "optimizer": args.optimizer,
        "model_file": args.model,
        "game_count": 0,
        "epoch_count": 1,
        "komi": args.komi
    }

    # check if optimizer is supported
    if metadata['optimizer'] != 'SGD':

        raise ValueError("Optimizer is not supported!")

    # create all directories
    # main folder
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # create weights file folder
    weight_folder = os.path.join(args.out_directory, FOLDER_WEIGHT)
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    # create sgf save file folder
    sgf_folder = os.path.join(args.out_directory, FOLDER_SGF)
    if not os.path.exists(sgf_folder):
        os.makedirs(sgf_folder)

    # save metadata
    metadata['meta_file'] = os.path.join(args.out_directory, FILE_METADATA)
    save_metadata( metadata )

    # hdf5 file location
    hdf5_file = os.path.join(args.out_directory, FILE_HDF5)

    # initialize at training start
    if args.verbose:
        print 'create initial model'

    # create and save initial network weights
    save_file = "weights.{version:05d}.hdf5".format(version=0)
    save_location = str( os.path.join(args.out_directory, FOLDER_WEIGHT, save_file) )
    worker = train_model( 0, metadata["model_file"], None, save_location, save_file, 0, metadata['meta_file'], None )
    worker.start()
    worker.join()

    # load metadata ( been changed by trainer )
    metadata = load_metadata( metadata['meta_file'] )

    # save initial settings
    metadata["best_model"]   = save_file
    metadata["newest_model"] = save_file

    # create hdf5 file
    HDF5_handler.create_hdf5_file( hdf5_file, metadata['feature_list'], metadata["board_size"], metadata["feature_depth"] )

    # save metadata   
    save_metadata( metadata )

    # start training
    run_training( metadata, args.out_directory, args.verbose )


def resume_training(args):
    """
       Read metadata file and resume training
    """

    # metadata json file location
    meta_file = os.path.join(args.out_directory, FILE_METADATA)

    # load data from json file
    metadata = load_metadata( meta_file )

    # TODO possible check if we need to train or validate new model

    # start training
    run_training( metadata, args.out_directory, args.verbose )

def train_epoch(args):
    """
       Read metadata file and train new models for #args['epochs'] epochs
    """

    # TODO train from scratch, train with certain weights?

    for _ in range( args['epochs'] ):

        # train and test new model
        print 'train model'


def handle_arguments( cmd_line_args=None ):
    """
       argument parser for start training, resume training and train epochs
    """

    parser = argparse.ArgumentParser(description='Generate self-play games and train network.')
    # subparser is always first argument
    subparsers = parser.add_subparsers(help='sub-command help')

    ########################################
    ############## sub parser start training
    train = subparsers.add_parser('start', help='Start generating self-play games and training.')  # noqa: E501

    ####################
    # required arguments
    train.add_argument("out_directory", help="directory where metadata and weights will be saved")  # noqa: E501
    train.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicyValue.save_model())")  # noqa: E501

    ####################
    # optional arguments
    train.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to start training with. Default: None", default=None)  # noqa: E501
    train.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    train.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: " + str(DEFAULT_BATCH_SIZE), type=int, default=DEFAULT_BATCH_SIZE)  # noqa: E501
    train.add_argument("--epoch-size", "-E", help="Amount of batches per epoch. Default: " + str(DEFAULT_EPOCH_SIZE), type=int, default=DEFAULT_EPOCH_SIZE)  # noqa: E501
    train.add_argument("--komi", "-k", help="Komi value (int). Default: " + str(DEFAULT_KOMI), type=float, default=DEFAULT_KOMI)  # noqa: E501
    train.add_argument("--test-amount", help="Amount of games to play to determine best model. Default: " + str(DEFAULT_TEST_AMOUNT), type=int, default=DEFAULT_TEST_AMOUNT)  # noqa: E501
    train.add_argument("--train-every", "-T", help="Train new model after this many games. Default: " + str(DEFAULT_TRAIN_EVERY), type=int, default=DEFAULT_TRAIN_EVERY)  # noqa: E501
    train.add_argument("--optimizer", "-O", help="Used optimizer. (SGD) Default: " + DEFAULT_OPTIMIZER, type=str, default=DEFAULT_OPTIMIZER)  # noqa: E501
    train.add_argument("--simulations", "-s", help="Amount of MCTS simulations per move. Default: " + str(DEFAULT_SIMULATIONS), type=int, default=DEFAULT_SIMULATIONS)  # noqa: E501
    train.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: " + str(DEFAULT_LEARNING_RATE), type=float, default=DEFAULT_LEARNING_RATE)  # noqa: E501
    train.add_argument("--resign-treshold", help="Resign treshold. Default: " + str(DEFAULT_RESIGN_TRESHOLD), type=float, default=DEFAULT_RESIGN_TRESHOLD)  # noqa: E501
    train.add_argument("--allow-resign", help="Percentage of games allowed to resign game. Default: " + str(DEFAULT_ALLOW_RESIGN), type=float, default=DEFAULT_ALLOW_RESIGN)  # noqa: E501

    # function to call when start training
    train.set_defaults(func=start_training)

    ########################################
    ############# sub parser resume training
    resume = subparsers.add_parser('resume', help='Resume generating self-play games and training. (Settings are loaded from savefile.)')  # noqa: E501

    ####################
    # required arguments
    resume.add_argument("out_directory", help="directory where metadata and weight files where stored during previous session.")  # noqa: E501

    ####################
    # optional arguments
    resume.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    # function to call when resume training
    resume.set_defaults(func=resume_training)


    ########################################
    ################# sub parser train epoch
    resume = subparsers.add_parser('train', help='Train model for x amount of epochs. (Settings are loaded from savefile.)')  # noqa: E501

    ####################
    # required arguments
    resume.add_argument("out_directory", help="directory where metadata and weight files where stored during previous session.")  # noqa: E501
    resume.add_argument("epochs", help="Amount of epochs to train.")  # noqa: E501

    ####################
    # optional arguments
    resume.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    # function to call when train epoch
    resume.set_defaults(func=train_epoch)

    # show help or parse arguments
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # execute function (start, resume or train)
    args.func(args)


if __name__ == '__main__':

    handle_arguments()
