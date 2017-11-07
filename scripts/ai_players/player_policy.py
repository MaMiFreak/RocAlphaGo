import sys
from AlphaGo.ai import ProbabilisticPolicyPlayer as Player
from AlphaGo.models.policy import CNNPolicy as Network
from interface.gtp_wrapper import run_gtp

# input to player should be
# model_file weight_file [ greedy_start ]
# greedy_start is optional
# TODO rewrite with argparse so it can take temperature, greedy_start, board_size and move_limit as well

greedy_start = 10000
temperature  = 1

# get files
MODEL   = sys.argv[ 1 ]
WEIGHTS = sys.argv[ 2 ]

# check if greedy start has been set
if len( sys.argv ) > 3:

    greedy_start = int( sys.argv[ 3 ] )
    # print( 'greedy start: ' + str( greedy_start ) )

# print( 'file model: ' + MODEL )
# print( 'file weights: ' + WEIGHTS )

# create network
network = Network.load_model( MODEL )
# load weights
network.model.load_weights( WEIGHTS )

# create ai player
player = Player( network, temperature=temperature, move_limit=10000, greedy_start=greedy_start )

# start gtp
run_gtp( player, board_size = 9 )

