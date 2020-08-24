import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

F=open ('./Raw/180315-run7-cs137_at_05_-05_11mm-2hrs-BigE-UnPhy-ComL_2x.csv', 'r')
outputfil = open('./Filtered/Cesium137_05_-05_11.csv', 'w')

MeCsq = 0.5109989461 # mass of electron in MeV

ue = 0.1 #Uncertainty of the energy measurements in MeV

# PUT IN THE ALLOWED ENERGIES PLEASE!!!
# Oxygen Prompts: 2.742, 5.240, 6.129, 6.916, 7.116
# Carbon: 4.444
# Nitrogen: 1.635, 2.313, 5.269, 5.298
# Boron: 0.718
# Cobalt60: 1.173, 1.332
# Cesium137: 0.6617
actual_E = np.array( [ 0.6617 ] )
CL = np.array( [ 1.04, 0.96 ] )

def uncertainty_triple_scatter(E1, E2, cosstheta2, ue ):
    u = 4.0*np.sqrt(abs((MeCsq**2*(1.0 + (E1**2*((-1.0 + cosstheta2)*E1**2 + 2.0*(-1.0 + cosstheta2)*E1*(E2 + np.sqrt(abs(E2*(E2 - (4.0*MeCsq)/(-1.0 + cosstheta2)))) ) + 2.0*E2*((-1.0 + cosstheta2)*E2 - 2.0*MeCsq + (-1.0 + cosstheta2)*np.sqrt(E2*(E2 - (4.0*MeCsq)/(-1.0 + cosstheta2))))))/(E2**3*((-1.0 + cosstheta2)*E2 - 4.0*MeCsq)))*ue**2)/ ((2.0*E1 + E2 + np.sqrt(abs(E2*(E2 - (4.0*MeCsq)/(-1.0 + cosstheta2))))  )**4*(1.0 - (1.0 - (4.0*E1*MeCsq)/((E2 + np.sqrt(abs(E2*(E2 - (4.0*MeCsq)/(-1.0 + cosstheta2))))  )*(2.0*E1 + E2 + np.sqrt(abs(E2*(E2 - (4.0*MeCsq)/(-1.0 + cosstheta2))))   )))**2))))

    if u > 0:
        return u
    else:
        return 10

def uncertainty_double_scatter(E1, E2, ue ):
    return np.sqrt( abs( ( ((E1**4) + 4 * (E1**3) * E2 + 4 * (E1**2) * (E2**2) + (E2**4)) * MeCsq * (ue**2)) / ( E1 * (E2**2) * (E1+E2)**2 * ( 2 * E2 * (E1+E2) - E1 * MeCsq ) ) ) )

def compton_rel( E0 , theta ):
    alpha = E0/MeCsq
    beta = alpha * ( 1 - np.cos(theta) )
    return E0 * beta / ( 1 + beta )

def unit_vector(start, stop):
  magsq = 0;
  vec = np.array(3)
  vec = stop - start
  if np.linalg.norm(vec) > 0:
      return vec/np.sqrt(np.dot(vec,vec))
  return np.zeros(3)

def angle_between (A, B):
  return np.abs( np.arccos( np.dot(A,B) ) );

E1_fil = []
E1_un = []
theta1_fil = []
theta1_un = []
E0_fil = []
E0_un = []

utheta1_un = []

filtered = 0

for line in F:
    line=line.strip()
    columns=line.split(",")

    E1 = float(columns[0])
    x1 = float(columns[1])
    y1 = float(columns[2])
    z1 = float(columns[3])
    E2 = float(columns[4])
    x2 = float(columns[5])
    y2 = float(columns[6])
    z2 = float(columns[7])

    axis1 = unit_vector( np.array([ x2, y2, z2 ]) , np.array([ x1, y1, z1 ]) )
    if np.linalg.norm(axis1)==0:
        continue

    if len(columns) <= 9:
        E0 = E1 + E2
        utheta1 = uncertainty_double_scatter( E1, E2, ue )

    else:
        E3 = float(columns[8])
        x3 = float(columns[9])
        y3 = float(columns[10])
        z3 = float(columns[11])

        axis2 = unit_vector( np.array([ x3, y3, z3 ]) , np.array([ x2, y2, z2 ]) )

        if np.linalg.norm(axis2)==0:
            continue

        theta2 = angle_between( axis1 , axis2 )
        E0 = E1 + ( E2 + np.sqrt( E2**2 + ( 4.0 * E2 * MeCsq ) / (1.0 - np.cos(theta2) ) ) ) / 2.0
        utheta1 = uncertainty_triple_scatter( E1, E2, np.cos(theta2), ue )



    if np.abs(1 + MeCsq * ( 1.0/(E0) - 1.0/(E0-E1) ) ) < 1 and math.isnan(utheta1)==False:
        theta1 = np.arccos( 1 + MeCsq * ( 1.0/(E0) - 1.0/(E0-E1) ) );

        E0_un.append( E0 )
        E1_un.append( E1 )
        theta1_un.append( theta1 )
        utheta1_un.append( utheta1 )

        for e in actual_E: # Passing the compton line filtering
            maxE1 = CL[0] * compton_rel( e , theta1)
            minE1 = CL[1] * compton_rel( e , theta1)

            if E1 > minE1 and E1 < maxE1 and utheta1 < np.pi/2:

                # We calculate the first part of the Klein-Nishina Coeficcient
                kn = 1 - E1/E0 + E0/(E0-E1)

                # Now all this is written to file, there are two different formats available

                outputfil.write( str(E1) + ',' + str(x1) + ',' + str(y1) + ',' + str(z1) + ',' + str(E2) + ',' + str(x2) + ',' + str(y2) + ',' + str(z2) + '\n' )
                # outputfil.write( str(x1) + ',' + str(y1) + ',' + str(z1) + ',' + str(x2) + ',' + str(y2) + ',' + str(z2) + ',' + str(theta1) + ',' + str(utheta1) + ',' + str(kn) + ',' + str(E1) + ',' + str(E2) + '\n' )


                E0_fil.append( E0 )
                E1_fil.append( E1 )
                theta1_fil.append( theta1 )

                filtered +=1

                break # Must not pass more than one filter as they might overlap and then the even is counted more than once

    if filtered == -1: # Incase we want to end the program early
        break
outputfil.close()

print "Filtered Events" , len(E1_fil)
print "Unfiltered Events" , len(E1_un)

conesblocks = math.floor( len(E1_fil) / 256 )
conestotal = conesblocks * 256
if conestotal == 0: conestotal = len(E1_fil)
GPUsize = 7.1e9
voxels = math.floor( (GPUsize/conestotal)**(1.0/3) )

print "Recomended Divisions for a Cube Space :  " , voxels
print "Recomended Cone Blocksize             :  " , int(conesblocks) , " x 256 = " ,  int(conestotal)

figh = plt.figure()
axh = figh.add_subplot(111)
y_hist, x_hist, _ = axh.hist( E0_un , 1000 , normed = 1 )
axh.set_xlim([0,6])
axh.set_xlabel('Calculated Energy (MeV)')
axh.yaxis.set_ticklabels([])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot( theta1_un , E1_un , ',' )
ax.plot( theta1_fil , E1_fil , ',' , color = 'red' )

ax.set_ylabel('Energy Lost at Fist Scatter (MeV)')
ax.set_xlabel('Scattering Angle (rad)')

angles = np.linspace( 0 , np.pi , 300 )

for e in actual_E:
    plt.plot( angles , compton_rel( e , angles) , 'k--', alpha = 0.5 )
    axh.plot( (e, e), (0, max(y_hist)), 'k--' )
    # plt.plot( angles , CL[0]*compton_rel( e , angles) , 'r--' )
    # plt.plot( angles , CL[1]*compton_rel( e , angles) , 'r--' )

plt.tight_layout()
plt.show()
