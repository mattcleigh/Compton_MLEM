import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x,y):
    return np.cos(x)*np.sin(y)

N = 1
limits = [ [0.0,np.pi/2] , [0,np.pi/2] ]


def Simpsons2D( N , limits ):
    integral = 0.0
    steps = [ (maxi-mini)/N for mini,maxi in limits ]

    for i in range(N):
        x = limits[0][0] + i*steps[0]

        for j in range(N):
            y = limits[1][0] + j*steps[1]

            integral += ( steps[0] * steps[1] / 36 ) * (
                            f( x, y ) +             4*  f( x+steps[0]/2, y ) +                 f( x+steps[0], y ) +
                         4* f( x, y+steps[1]/2 ) +  16* f( x+steps[0]/2, y+steps[1]/2 ) +   4* f( x+steps[0], y+steps[1]/2 ) +
                            f( x, y+steps[1] ) +    4*  f( x+steps[0]/2, y+steps[1] ) +        f( x+steps[0], y+steps[1] ) )
    return integral


for N in range(1,100):
    print N, Simpsons2D( N , limits )
