# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:35:35 2021

@author: lpama
"""

from astropy.table import Table
import smatch
import math as mt
import numpy as np
import seaborn as sns
import pylab as plt
import pandas as pd
from matplotlib.pyplot import errorbar
from scipy import optimize

dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()


dat2= Table.read('2mass.fit', format='fits')
df2 = dat.to_pandas()



ra1 =dat.field('ALPHA_J2000')
dec1 = dat.field('DELTA_J2000')
mag1 = dat.field('MAG_AUTO') 
mag1_error = dat.field('MAGERR_AUTO') 
ra2 = dat2.field('RAJ2000')
dec2 = dat2.field('DEJ2000')
kmag2 = dat2.field('Kmag')
error_kmag2 = dat2.field('e_Kmag')

nside=4096 # healpix nside
maxmatch=1 # return closest match
radius= 1/3600.

# ra,dec,radius in degrees
matches = smatch.match(ra1, dec1, radius, ra2, dec2, nside=nside, maxmatch=maxmatch)
#print (matches)

ra1matched  = ra1[ matches['i1'] ]
dec1matched = dec1[ matches['i1'] ]
mag1_matched = mag1[ matches['i1'] ]
mag1_error_matched = mag1_error[ matches['i1'] ]
ra2matched  = ra2[ matches['i2'] ]
dec2matched = dec2[ matches['i2'] ]
kmag2_matched = kmag2[ matches['i2'] ]
error_kmag2_matched = error_kmag2[ matches['i2'] ]


cosgamma=[]
for i in range(len(matches)):
    gamma= mt.cos(90-dec1matched[i])*mt.cos(90-dec2matched[i])+mt.sin(90-dec1matched[i])*mt.sin(90-dec2matched[i])*mt.cos(ra1matched[i]-ra2matched[i])
    cosgamma.append(mt.acos(gamma)*3600)



#sns.distplot(cosgamma, kde=False)

#print(np.max(cosgamma))

#result= sns.jointplot(x=kmag2matched, y=mag1matched, kind="reg", truncate=False, xlim=(5, 25), ylim=(5, 25), color="m", height=7)
#plt.plot(kmag2matched, mag1matched, ".")
#print (result)

mask = kmag2_matched>12.3


kmag2_mask = kmag2_matched[mask]

mag1_mask = mag1_matched[mask]

#Hacer error final aplicando la mascara
error_kmag2_mask =  error_kmag2_matched[mask]

error_mag1_mask =  mag1_error_matched[mask]

erro_kmag2_mask = error_kmag2_matched[mask]

p = plt.polyfit(mag1_mask, kmag2_mask, 1)
print(p[0], p[1])

kmag2_mask_ajuste= p[0]*mag1_mask + p[1]



'''
plt.plot(kmagFinal, mag1Finalajuste) #, kmagFinal, mag1Finalajuste1)
plt.plot([10,20],[10,20])


plt.plot(kmag2matched, mag1matched-p[1], ".")

plt.legend(('Datos sin calibrar', 'Ajuste', 'y=x', 'Datos calibrados'))
plt.xlabel('KMAG')
plt.ylabel('MAG_AUTO')
plt.show()
#p2= plt.polyfit(kmagFinal, mag1Finalajuste1, 1)

#print (p2[0], p2[1])
dat['MAG_AUTO_CORRECTED'] = mag1-p[1]

dat.write('Sharks_sgp_e_2_cat_small.fits', overwrite=True)
'''





#kmagFinal, mag1Final

# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

#Poner aqui el error en leastsq

pinit = [p[0], p[1]]
out = optimize.leastsq(errfunc, pinit, args=(mag1_mask, kmag2_mask, error_mag1_mask), full_output=1)


pfinal = out[0]
covar = out[1]
print (pfinal)

print (error_kmag2_mask)

error_kmag2_mask= pfinal[1]*mag1_mask + pfinal[0]

plt.clf()
plt.subplot(1, 1, 1)
#plt.plot(kmag2matched, powerlaw(kmag2matched, amp, index))     # Fit
plt.plot(mag1_mask, kmag2_mask_ajuste) # Azul
plt.plot(mag1_mask, error_kmag2_mask) # Naranja
plt.errorbar(mag1_mask, kmag2_mask, yerr=error_kmag2_mask, fmt='k.')  # Data
plt.legend(('Ajuste lineal', 'Ajuste lineal con error', 'Valores con su error'))
plt.title('Best Fit')
plt.ylabel('KMAG (2MASS)')
plt.xlabel('MAG_AUTO (SHARKS)')