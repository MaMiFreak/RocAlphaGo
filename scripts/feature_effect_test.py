import numpy as np
from random import randint
import matplotlib.pyplot as plt
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
np.set_printoptions(threshold=np.nan)
from AlphaGo.preprocessing.preprocessing import Preprocess

# settings
versions   = 1 + 117
test_every = 10
amount     = 10
folder     = 'feature_tests/samples_' + str( amount ) + '/'
MODEL      = 'policy_all_192.json'
policy     = CNNPolicy.load_model( MODEL )

setting_pairs = [ [ "features",         3 , 48 ],
                  [ "board",            0 ,  3 ],
                  [ "ones",             3 ,  4 ],
                  [ "turns_since",      4 , 12 ],
                  [ "liberty",         12 , 20 ],
                  [ "capture_size",    20 , 28 ],
                  [ "self_atari_size", 28 , 36 ],
                  [ "liberty_after",   36 , 44 ],
                  [ "ladder_capture",  44 , 45 ],
                  [ "ladder_escape",   45 , 46 ],
                  [ "sensible",        46 , 47 ],
                  [ "zeros",           47 , 48 ]
                ]

setting_pairs = [ [ "features",         3 , 48 ],
                  [ "board",            0 ,  3 ],
                  [ "ones",             3 ,  4 ],
                  [ "turns_since",      4 , 12 ],
                  [ "liberty",         12 , 20 ],
                  [ "capture_size",    20 , 28 ],
                  [ "self_atari_size", 28 , 36 ],
                  [ "liberty_after",   36 , 44 ],
                  [ "ladder_capture",  44 , 45 ],
                  [ "ladder_escape",   45 , 46 ],
                  [ "sensible",        46 , 47 ]
                ]

positions = []
data      = {}
results   = {}
models    = []

# add leading zeros to a number string
def add_leading_zeros( value ):
    ret = str( value )
    while( len( ret ) < 5 ):

        ret = '0' + ret
    return ret

# calculate error
# TODO find best way to compare
def calculate_mse( model, inputs, base ):

    mse = 0.0
    per = 0.0

    for x in range( len( inputs ) ):

        prediction = model.forward( inputs[ x ] )

        difference = prediction - base[ x ]
        difference = ( difference / base[ x ] )
        per        = per + difference.sum()

        value = ( ( prediction - base[ x ] ) ** 2 ).sum()
        mse   = mse + value

    mse = mse / float( len( inputs ) )
    per = per / float( len( inputs ) )

    return mse, per

# duplicate numpy arrays and remove certain feature layers
def remove_features( samples, layer_min, layer_max ):

    # return array
    array = []
    count = 0

    # loop over all samples
    for sample in samples:

        # duplicate numpy array
        dup = np.copy( sample )

        # loop over the layers to be removed
        for z in range( layer_min, layer_max ):

            # loop over x and y and set value to 0
            for x in range( 19 ):

                for y in range( 19 ):

                    if dup[ 0 ][ z ][ x ][ y ] != 0:

                        count = count + 1
                    dup[ 0 ][ z ][ x ][ y ] = 0

        # add new sample
        array.append( dup )

    print "zeroed: " + str( count )

    return array, count

def generate_image( location, val_x, list_val_y ):

    fig     = plt.figure( figsize = ( 15, 6 ), dpi = 360 )
    legenda = []

    # loop over all values in list_val_y
    for val_y in list_val_y:

        line, = plt.plot( val_x, val_y[ 1 ], label = val_y[ 0 ] )
        legenda.append( line )

    plt.grid()
    plt.legend( handles = legenda, loc = 'lower right' )
    plt.savefig( location + '.png' )
    plt.clf()

# generate positions
for i in range( amount ):

    # new gamestate
    state = GameState()

    # do 50 up to 100 random moves 
    for move in range( randint( 50, 100 ) ):

        moves  = state.get_legal_moves(include_eyes=False)
        choice = np.random.choice( len( moves ) )
        state.do_move( moves[ choice ] )

    # add preprocessed position
    positions.append( policy.preprocessor.state_to_tensor( state ) )

print str( amount ) + ' positions generated'

# generate all zeroed input arrays and initialize result lists
for setting in setting_pairs:

    input_without, count = remove_features( positions, setting[ 1 ], setting[ 2 ] )
    data[ setting[ 0 ] ] = [ input_without, count ]

    results[ setting[ 0 ] + "mse_nor" ] = []
    results[ setting[ 0 ] + "mse_div" ] = []
    results[ setting[ 0 ] + "per_nor" ] = []
    results[ setting[ 0 ] + "per_div" ] = []

print 'all inputs zeroed'

# generate list with models to be tested
for i in range( versions ):

    if i % test_every == 0:

        models.append( i )

# loop over all models and calculate mse of each sample
for model in models:

    WEIGHTS = 'model/weights.' + add_leading_zeros( model ) + '.hdf5'
    policy.model.load_weights( WEIGHTS )

    # calculate prediction with original data
    prediction_base = []
    for sample in positions:

        prediction_base.append( policy.forward( sample ) )


    # loop over all zeroed settings
    for setting in setting_pairs:

        # calculate prediction with zeroed data

        mse, per = calculate_mse( policy, data[ setting[ 0 ] ][ 0 ], prediction_base )
        results[ setting[ 0 ] + "mse_nor" ].append( mse )
        results[ setting[ 0 ] + "mse_div" ].append( mse / data[ setting[ 0 ] ][ 1 ] )

        results[ setting[ 0 ] + "per_nor" ].append( per )
        results[ setting[ 0 ] + "per_div" ].append( per / data[ setting[ 0 ] ][ 1 ] )

        print WEIGHTS + ' model ready ' + setting[ 0 ]
    print WEIGHTS + ' model ready'

# generate images
options = [ "mse_nor", "mse_div", "per_nor", "per_div" ]

for var in options:

    mix_arr = []

    # generate image with all tests
    for setting in setting_pairs:

        mix_arr.append( [ setting[ 0 ], results[ setting[ 0 ] + var ] ] )

    generate_image( folder + 'all_' + var, models, mix_arr )

    # generate images per test
    for setting in setting_pairs:

        name = folder + var + '_remove_' + str( setting[ 1 ] ) + '_' + str( setting[ 2 ] ) + '_' + setting[ 0 ]
        generate_image( name , models, [ [ setting[ 0 ], results[ setting[ 0 ] + var ] ] ] )

