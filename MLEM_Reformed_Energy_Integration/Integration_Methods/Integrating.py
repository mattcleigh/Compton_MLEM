import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

max_steps = 30
sig_val = 3.0


sig_actual_val = 0.987580669348448
sig_limit = 2.5

def Gaussian( x , sigma ):
    return np.exp( (-x*x)/(2*sigma*sigma) ) / ( np.sqrt(2*np.pi) * sigma )

def Trapezoidal( start , stop , sigma , int_steps):
    integral = 0.0
    step = (stop-start)/int_steps
    for i in range(int_steps):
        pos = start + i * step
        integral += ( Gaussian( pos, sigma) +  Gaussian( pos+step, sigma) ) * step / 2
    return integral

def Simpson( start , stop , sigma , int_steps):
    integral = 0.0
    step = (stop-start)/int_steps
    for i in range(int_steps):
        pos = start + i * step
        integral += ( Gaussian( pos, sigma) + 4*Gaussian( pos+float(step)/2, sigma) + Gaussian( pos+step, sigma) ) * step / 6
    return integral

methods = [ Trapezoidal , Simpson ]
labels = [ "Trapezoidal" , "Simpson" ]


integration_steps = np.arange(1,max_steps+1)

fig = plt.figure( figsize = (4,4) )
ax = fig.add_subplot( 1 , 1 , 1 )
ax.set_xlabel('Number of Divisions')
ax.set_ylabel('Computation Integral Value')

for f in range(len(methods)):
    function = methods[f]
    totals = np.zeros(max_steps)

    for i in range(max_steps):
        divi = integration_steps[i]
        totals[i] = function( -sig_limit*sig_val , sig_limit*sig_val , sig_val , divi ) / sig_actual_val

    ax.plot( integration_steps , totals, 'x', label = labels[f] )

    print totals

ax.plot( [0,max_steps] , [1,1] , 'k--' )

plt.tight_layout()
plt.legend()
plt.show()
