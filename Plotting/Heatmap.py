import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import sys

name = "180315-run7-cs137_at_05_-05_11mm-2hrs-BigE-UnPhy-ComL_2x"

XDIVI = 20
YDIVI = 20
ZDIVI = 20

CONES = 57344

x_start, x_end = -50, 50
y_start, y_end = -50, 50
z_start, z_end = -50, 50

TotalIt = 20

SAVEEVERY = 10

plotlist = [0,1]
plotlist += range( SAVEEVERY , TotalIt + SAVEEVERY , SAVEEVERY )

svname = name + "_C" + str(CONES) + "_x" + str(XDIVI) + "y" + str(YDIVI) + "z" + str(ZDIVI)

for It in plotlist:
    file_name = svname + "_I" + str(It)

    F=open ('../Output/' + file_name + '.csv', 'r')

    fxz = [[ 0 for i in range(XDIVI) ]  for k in range(ZDIVI)]
    fxy = [[ 0 for i in range(XDIVI) ]  for j in range(YDIVI)]
    fyz = [[ 0 for j in range(YDIVI) ]  for k in range(ZDIVI)]

    for line in F:
        line=line.strip()
        columns=line.split(",")
        i = int(columns[4])
        j = int(columns[5])
        k = int(columns[6])

        fxz[k][i] += float(columns[0])
        fxy[j][i] += float(columns[0])
        fyz[k][j] += float(columns[0])

    fig = plt.figure( figsize = (6,6) )
    ax = fig.add_subplot( 1,1,1 )
    ax.set_title(' XZ-Heatmap' )
    ax.set_xlabel('x direction (mm)')
    ax.set_ylabel('z direction (mm)')
    # ax.set_title( str(It) )
    heatmap = ax.imshow(fxy, cmap = 'jet', origin = 'lower', extent = [x_start,x_end,y_start,y_end], interpolation='none', norm=LogNorm() )

    # cbar = fig.colorbar(heatmap)
    plt.tight_layout()
    # plt.show()
    fig.savefig('../Images/Heatmap' + file_name + '.png')
    sys.stdout.write("Images Completed: %d   \r" % (It) )
    sys.stdout.flush()
    plt.close(fig)




















###
