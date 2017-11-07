import os
import json
import codecs
import os.path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# root folder
net_name       = 'resnet_96_20'
root           = 'test_results_' + net_name + '/'
name           = 'test_result'
info           = name + '.html'
min_update     = 0
min_version    = 0
versions       = 1 + 4
greedy_start   = '8'
leela_playouts = [ '100' ] # [ '100', '1000', '10000' ]

legenda        = [] 

def get_html_data( data ):
    """
       twogtp -analyze function creates html with win ratio
       find winratio in this file and return as float
    """

    # get black win ratio value
    search_start   = '<tr><th align="left" valign="top" nowrap>Black wins:</th><td align="left">'
    location_start = data.find( search_start )
    search_end     = '%'
    location_end   = data.find( search_end, location_start )
    black_win      = data[ location_start + len( search_start ) : location_end ]

    return float( black_win )

def read_folders( dat, html, version_min, version_max, increment, force_html ):
    """
       loop over all folders, read twogtp html files and return winratios
    """

    val_x = []
    val_y = []

    for i in range( version_min, version_max, increment ):

        file_dat  = dat.format( str( i ) ) + '.dat'
        file_html = html.format( str( i ) )

        if i >= force_html and os.path.isfile( file_dat ):
            # update html
            command = "gogui-twogtp -analyze " + file_dat + " -force"
            os.system( command )

        if os.path.isfile( file_html ):

            # read file
            reader = codecs.open( file_html, 'r' )
            data   = reader.read()

            value_black_win = get_html_data( data )
            val_y.append( ( 100 - value_black_win ) / 100 ) 
            val_x.append( i )

    return val_x, val_y

########################################
########################### create image

fig, ax = plt.subplots(figsize=(15, 6), dpi=360)
plt.title( net_name + ' policy model training progress', fontsize=14, fontweight='bold' )

# Twin the x-axis twice to make independent y-axes for win ratio and training accuracy
axes = [ ax, ax.twinx() ]
axes[0].set_ylabel( 'vs opponent win ratio', color = 'blue' )
axes[0].tick_params(axis='y', colors = 'blue' )
axes[0].set_xlabel( 'epoch' )
axes[0].set_ylim([0,1])
minorLocator = MultipleLocator(0.1)
axes[0].yaxis.set_minor_locator(minorLocator)

axes[1].set_ylabel( 'accuracy', color = 'orange' )
axes[1].tick_params(axis='y', colors = 'orange' )
axes[1].set_ylim([0.4,0.65])

########################################
########################### vs player 96

folder  = root + "player_96_41/96f_v41_vs_" + net_name + "_v{}/"
command = 'vs 96f v41 (100 games - probabilistic)'

val_x, val_y = read_folders( folder + name, folder + info, min_version, versions, 1, min_update )

print 'player 96'
print val_y

line, = axes[0].plot( val_x, val_y, marker='o', label= command )
legenda.append(line)

########################################
########################## vs player 192

folder  = root + "player_192_132/192f_v132_vs_" + net_name + "_v{}/"
command = 'vs 192f v132 (100 games - probabilistic)'

val_x, val_y = read_folders( folder + name, folder + info, min_version, versions, 1, min_update )

print 'player 192'
print val_y

line, = axes[0].plot( val_x, val_y, marker='o', label= command )
legenda.append(line)


########################################
############################### vs leela

for leela_playout in leela_playouts:

    folder  = root + "leela_" + leela_playout + "/leela_vs_" + net_name + "_v{}_gr_" + greedy_start + "/"
    command = 'vs leela (100 games - greedy ' + greedy_start + ')[leela_0100_linux_x64 --gtp -p ' + leela_playout + ' --noponder]'

    val_x, val_y = read_folders( folder + name, folder + info, 0, versions, 1, min_update )

    print 'leela'
    print val_y

    line, = axes[0].plot( val_x, val_y, marker='o', label= command )
    legenda.append(line)

########################################
############################### accuracy

file_location = root + 'metadata_policy_supervised.json'

with open(file_location, 'r') as f:

    metadata = json.load(f)

    train_acc = []

    for ep in metadata['epoch_logs']:

        if len( train_acc ) < versions:

            train_acc.append( ep['acc'] )

    #[ ep['acc'] for ep in metadata['epoch_logs']]
    line, = axes[1].plot( range( len( train_acc ) ), train_acc, color = 'orange', label= 'training accuracy' )
    legenda.append(line)

########################################
############################# save image

axes[0].grid(which='both')
#plt.legend(handles=legenda, loc='lower right')
plt.legend(handles=legenda, loc='upper left')
plt.savefig( 'training_progress_' + net_name + '.png' )





