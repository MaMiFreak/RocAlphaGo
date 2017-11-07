import os
import json
import codecs
import os.path
import matplotlib.pyplot as plt

net_list    = [ 'policy_96', 'policy_96_s2', 'policy_192', 'resnet_64_20', 'resnet_96_20', 'resnet_128_20' ]
net_list    = [ 'policy_96', 'policy_192', 'resnet_64_20', 'resnet_96_20', 'resnet_128_20' ]
version_min = 0
version_max = 15
max_xval    = 40000000

legenda     = [] 

########################################
########################### create image

fig, ax = plt.subplots(figsize=(15, 6), dpi=360)
plt.title( 'Model comparison', fontsize=14, fontweight='bold' )

########################################
##### loop over all models and draw data

for model in net_list:

    file_location = 'test_results_' + model + '/metadata_policy_supervised.json'

    with open(file_location, 'r') as f:

        metadata = json.load(f)

        train_y = []
        train_x = []

        x = 0
        for ep in metadata['epoch_logs']:

            xval = ( x + 1 ) * metadata['epoch_length']

            if x >= version_min and x <= version_max and xval < max_xval:

                train_y.append( ep['acc'] )
                train_x.append( xval )
                #train_x.append( x )
                x = x + 1

        #[ ep['acc'] for ep in metadata['epoch_logs']]
        line, = plt.plot( train_x, train_y, label= model )
        legenda.append(line)

########################################
############################# save image

plt.legend(handles=legenda, loc='lower right')
plt.savefig( 'models_accuracy_comparison.png' )
