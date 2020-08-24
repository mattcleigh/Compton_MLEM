import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# This program is useful for displaying the estimated location of a point source and comparing it to an actual location
# It takes an output file and searches for the brightest voxel in the space, once the voxel is found the 3D coordinates are calculated
# 3 heatmaps and 3 profiles are then created along those coordinates, the actual location of the source is displayed in these as well
# The various profiles are also exported as csv files and a textfile is created showing the distances from the estimated and actual locations

name = "180315-run7-cs137_at_05_-05_11mm-2hrs-BigE-UnPhy-ComL_2x_d1"

XDIVI = 100
YDIVI = 100
ZDIVI = 100

CONES = 20000

x_start, x_end = -50, 50
y_start, y_end = -50, 50
z_start, z_end = -50, 50

It = 150

actual_locations = np.array( [ 5 , -5 , 11 ] )

########################################################################################
divis = np.array ( ( XDIVI , YDIVI , ZDIVI ) , dtype = int )
limits = np.array ( ( [x_start, x_end] , [y_start, y_end] , [z_start, z_end] ) , dtype = float )
deltas = [ (limit[1] - limit[0])/div for limit, div in zip(limits, divis) ]
letters = [ 'x' , 'y' , 'z' ]

f_max = 0.0
max_locs = np.zeros(3)
max_pos = np.zeros(3)

open_name  = name + "_C" + str(CONES) + "_x" + str(XDIVI) + "y" + str(YDIVI) + "z" + str(ZDIVI) + "_I" + str(It)
F=open ( '../Output/' + open_name + '.csv' , 'r')
for line in F:
    line=line.strip()
    columns=line.split(",")
    f = float(columns[0])
    locs = [ int(columns[4]) , int(columns[5]) , int(columns[6]) ]

    distance_to_corner = min ( min( ( ( divis - 1 ) - locs ) ) ,  min(locs) )

    if f > f_max and distance_to_corner > np.mean(divis)/50:
        f_max = f
        max_locs = np.array([ int(columns[4]) , int(columns[5]) , int(columns[6]) ])
        max_pos = np.array([ float(columns[1]) , float(columns[2]) , float(columns[3]) ])

difference = np.subtract( max_pos , actual_locations )
magn = np.linalg.norm(difference)

txt = open('../Images/Textfile_' + open_name + '.txt','w')
txt.write( "Actual Location of source:     {:06.2f} mm   ,   {:06.2f} mm   ,   {:06.2f} mm \n".format( actual_locations[0] , actual_locations[1] , actual_locations[2] ) )
txt.write( "Estimated Location of source:  {:06.2f} mm   ,   {:06.2f} mm   ,   {:06.2f} mm \n".format( max_pos[0] , max_pos[1] , max_pos[2] ) )
txt.write( "                                    _______________________________\n" )
txt.write( "Difference:                    {:06.2f} mm   ,   {:06.2f} mm   ,   {:06.2f} mm \n".format( difference[0] , difference[1] , difference[2] ) )
txt.write( "Magnitude:                     {:06.2f} mm \n".format( magn ) )
txt.write( "\n \n \nResolution (voxel-size): {:.2f} x {:.2f} x {:.2f} mm".format( deltas[0] , deltas[1] , deltas[2] ) )
txt.close()

# Printing the slides
for i in range(3):


    ri = np.delete ( ([0,1,2]) , i )
    act = actual_locations[i]
    position = max_pos[i]

    prof = open( '../Images/{}-profile_at_{}={:.2f}_and_{}={:.2f}.csv'.format( letters[i] , letters[ ri[0] ] , max_pos[ ri[0] ] , letters[ ri[1] ] , max_pos[ ri[1] ] ) , 'w' )

    f_grid = [[ 0 for a in range(divis[ri[0]]) ]  for b in range(divis[ri[1]])]
    f_profile = np.array( [ 0.0 for c in range(divis[i]) ] )

    F=open ( '../Output/' + open_name + '.csv' , 'r')
    for line in F:
        line=line.strip()
        columns=line.split(",")
        current_locs = [ int(columns[4]) , int(columns[5]) , int(columns[6]) ]

        if current_locs[i] == max_locs[i]:
            f_grid[ current_locs[ ri[1] ] ][ current_locs[ ri[0] ] ] += float( columns[0] )

        if np.array_equal ( np.delete ( max_locs , i ) , np.delete ( current_locs , i ) ):
            f_profile[ current_locs[i] ] += float( columns[0] )
            prof.write( " {:6} , {:6} , {:6} \n".format( columns[0] , columns[ i+1 ] , columns[ i+4 ] ) )

    prof.close()

    fig = plt.figure( figsize = (6,6) )
    ax = fig.add_subplot( 1,1,1 )
    ax.set_xlabel( letters[ ri[0] ] + ' direction (mm)')
    ax.set_ylabel( letters[ ri[1] ] + ' direction (mm)')
    ax.set_title( letters[i] + ' = {:.2f} mm'.format(position)  )
    heatmap = ax.imshow(f_grid, cmap = 'jet', origin = 'lower', extent = [ limits[ri[0]][0] , limits[ri[0]][1] , limits[ri[1]][0] , limits[ri[1]][1] ], interpolation='gaussian', vmin = 0, vmax = f_max )
    actual = ax.scatter( actual_locations[ ri[0] ] , actual_locations[ ri[1] ] , s=150 , c = 'white' , marker = 'x' )
    plt.tight_layout()

    figp = plt.figure( figsize = (6,6) )
    axp = figp.add_subplot( 1,1,1 )
    axp.set_xlabel( letters[ i ] + ' direction (mm)')
    axp.set_ylabel( ' % of maximum')
    space = np.linspace( limits[i][0] , limits[i][1] , divis[i] )

    axp.set_title( '{}-profile at {} = {:.2f} mm , and {} = {:.2f} mm'.format( letters[i] , letters[ ri[0] ] , max_pos[ ri[0] ] , letters[ ri[1] ] , max_pos[ ri[1] ] ) )
    axp.step( space , 100*f_profile/max(f_profile) , where = 'mid', label = "Estimated = {:.2f} mm".format(max_pos[i]) )
    axp.plot( [act,act] , [0,100], 'k--', label = "Actual = {:.2f} mm".format(act) )

    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig('../Images/Slide_' + letters[i] + "_" + open_name + '.png')
    figp.savefig('../Images/Profile_' + letters[i] + "_" + open_name + '.png')
    plt.close(fig)
    plt.close(figp)

















#
