import os
net_name = 'resnet_96_20'
base     = 'test_results_' + net_name   # base folder to store data
amounts  = [ '1', '100', '200', '400' ] # amount of games to play
versions = 1 + 4                        # max epoch number of new network
min_ver  = 0                            # min epoch number of network 
name     = 'test_result'                # twogtp .dat filename

def play_games( file_dat, dir_name, test_amount, opponent, player, extra_commands ):
    """
       create folder and let twogtp generate #test_amount games with opponent vs player
       optional: extra_commands for twogtp
    """

    # create folder
    directory = os.path.dirname( dir_name )

    # create directory
    if not os.path.exists( directory ):
        os.makedirs( directory )

    # run games
    command = 'gogui-twogtp -black "' + opponent + '" -white "' + player + \
              '" -auto -sgffile ' + file_dat + ' -games ' + test_amount + ' -alternate -referee "gnugo --mode gtp --chinese-rules"'
    os.system( command )

    # create info html
    command = "gogui-twogtp -analyze " + file_dat + '.dat' + " -force"
    os.system( command )


# loop over all test amounts
for amount in amounts:

    ########################################
    ########################### vs player 96

    # settings
    root     = base + '/player_96_41/96f_v41_vs_' + net_name + '_v'
    vs       = 'python player_96.py 41' # opponent player

    # play all players vs 96f v41 policy net
    for i in reversed( range( min_ver, versions ) ):
    
        # create folder
        dir_name  = root + str( i ) + "/"

        play_games( dir_name + name, dir_name, amount, vs, 'python player_' + net_name + '.py ' + str(i), '' )

        print 'v' + str( i ) + ' vs 96f v41 policy player ' + amount + ' games ready'

    ########################################
    ########################## vs player 192

    # settings
    root     = base + '/player_192_132/192f_v132_vs_' + net_name +'_v'
    vs       = 'python player_192.py 132' # opponent player

    # play all players vs 192f v132 policy net
    for i in reversed( range( min_ver, versions ) ):
    
        # create folder
        dir_name  = root + str( i ) + "/"

        play_games( dir_name + name, dir_name, amount, vs, 'python player_' + net_name + '.py ' + str(i), '' )

        print 'v' + str( i ) + ' vs 192f v132 policy player ' + amount + ' games ready'

    ########################################
    ############################### vs leela

    # settings
    greedy_start = '8'
    playouts     = [ '100' ] # [ '100', '1000', '10000' ]

    # loop over all leela playout settings
    for leela_playout in playouts:

        root  = base + '/leela_' + leela_playout + '/leela_vs_' + net_name + '_v'
        vs = './leela/leela_0100_linux_x64 --gtp -p ' + leela_playout + ' --noponder' # opponent player

        # play all players vs leela
        for i in reversed( range( min_ver, versions ) ):
    
            # create folder
            dir_name  = root + str( i ) + '_gr_' + greedy_start + "/"

            play_games( dir_name + name, dir_name, amount, vs, 'python player_' + net_name + '.py ' + str(i) + ' ' + greedy_start, '' )

            print 'v' + str( i ) + ' vs leela ' + amount + ' games ready'











