# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:52:28 2021

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
from scipy import optimize, stats

dat = Table.read('Sharks_sgpe.fits', format='fits')
df = dat.to_pandas()


dat2= Table.read('2mass_in_field.fits', format='fits')
df2 = dat2.to_pandas()



ra_sharks =dat.field('RA')
dec_sharks = dat.field('DEC')
magnitud_sharks = dat.field('APERMAG3') 
magnitud_error_sharks = dat.field('APERMAG3ERR') 

ra_2mass = dat2.field('RAJ2000')
dec_2mass = dat2.field('DEJ2000')
magnitud_2mass = dat2.field('Kmag')
magnitud_error_2mass = dat2.field('e_Kmag')

nside=4096 # healpix nside
maxmatch=1 # return closest match
radius= 1/3600.

# ra,dec,radius in degrees
matches = smatch.match(ra_sharks, dec_sharks, radius, ra_2mass, dec_2mass, nside=nside, maxmatch=maxmatch)
#print (matches)

ra_sharks_matched  = ra_sharks[ matches['i1'] ]
dec_sharks_matched = dec_sharks[ matches['i1'] ]
magnitud_sharks_matched = magnitud_sharks[ matches['i1'] ]
magnitud_error_sharks_matched = magnitud_error_sharks[ matches['i1'] ]

ra_2mass_matched  = ra_2mass[ matches['i2'] ]
dec_2mass_matched = dec_2mass[ matches['i2'] ]
magnitud_2mass_matched = magnitud_2mass[ matches['i2'] ]
magnitud_error_2mass_matched = magnitud_error_2mass[ matches['i2'] ]


cosgamma=[]
for i in range(len(matches)):
    gamma= mt.cos(90-dec_sharks_matched[i])*mt.cos(90-dec_2mass_matched[i])+mt.sin(90-dec_sharks_matched[i])*mt.sin(90-dec_2mass_matched[i])*mt.cos(ra_sharks_matched[i]-ra_2mass_matched [i])
    cosgamma.append(mt.acos(gamma)*3600)



#sns.distplot(cosgamma, kde=False)

#print(np.max(cosgamma))

#result= sns.jointplot(x=magnitud_2mass_matched, y=magnitud_sharks_matched, kind="reg", truncate=False, xlim=(5, 25), ylim=(5, 25), color="m", height=7)
#plt.plot(kmag2matched, mag1matched, ".")
#print (result)

mask = (magnitud_2mass_matched>12.3)&(magnitud_error_2mass_matched>0)


magnitud_2mass_mask = magnitud_2mass_matched[mask]

magnitud_sharks_mask = magnitud_sharks_matched[mask]

#Hacer error final aplicando la mascara
magnitud_error_2mass_mask =  magnitud_error_2mass_matched[mask]

magnitud_error_sharks_mask =  magnitud_error_sharks_matched[mask]



p = plt.polyfit(magnitud_2mass_mask, magnitud_sharks_mask, 1)
print(p[0], p[1])

magnitud_sharks_mask_ajuste= p[0]*magnitud_2mass_mask + p[1]




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
out = optimize.leastsq(errfunc, pinit, args=(magnitud_2mass_mask, magnitud_sharks_mask,  magnitud_error_2mass_mask), full_output=1)


pfinal = out[0]
covar = out[1]
print (pfinal)

ajuste_con_error_magnitud_sharks_mask = pfinal[1] * magnitud_2mass_mask + pfinal[0]

plt.subplot(1, 1, 1)
plt.plot(magnitud_2mass_mask, magnitud_sharks_mask_ajuste) # Azul
plt.plot(magnitud_2mass_mask, ajuste_con_error_magnitud_sharks_mask) # Naranja
plt.errorbar(magnitud_2mass_mask, magnitud_sharks_mask, yerr=magnitud_error_2mass_mask, fmt='k.')  # Data
plt.legend(('Ajuste lineal', 'Ajuste lineal con error', 'Valores con su error'))
plt.title('Best Fit')
plt.xlabel('KMAG (2MASS)')
plt.ylabel('MAG_AUTO (SHARKS)')
plt.savefig('ajuste_con_error.png')


plt.clf()

z = (magnitud_sharks_mask - ajuste_con_error_magnitud_sharks_mask )/magnitud_error_2mass_mask

pf = pd.DataFrame(zip(magnitud_sharks_mask, magnitud_2mass_mask, magnitud_error_2mass_mask))

pf = pf[(np.abs(z) < 2.5)]
#pf = pf[(np.abs(stats.zscore(pf)) < 2.5).all(axis=1)]
print(len(magnitud_sharks_mask), len(pf))
m, b = plt.polyfit(pf[1], pf[0], 1)

print(b, m)
#_ = plt.plot(pf[1], pf[0], 'o', label='Original data', markersize=2)
_ = plt.plot(pf[1], m*pf[1] + b, 'r', label='Fitted line')
_ = plt.errorbar(pf[1], pf[0], yerr=pf[2], fmt='k.', label='Original data with its error')
_ = plt.legend()
plt.savefig('ajuste_sin_outliers.png')

plt.show()


plt.clf()