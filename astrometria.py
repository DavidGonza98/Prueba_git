# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:04:55 2021

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

tips= [df, df2]

ra1 =dat.field('ALPHA_J2000')
dec1 = dat.field('DELTA_J2000')
mag1 = dat.field('MAG_AUTO') 
magger1 = dat.field('MAGERR_AUTO') 
ra2 = dat2.field('RAJ2000')
dec2 = dat2.field('DEJ2000')
kmag2 = dat2.field('Kmag')
ekmag2 = dat2.field('e_Kmag')

nside=4096 # healpix nside
maxmatch=1 # return closest match
radius= 1/3600.

# ra,dec,radius in degrees
matches = smatch.match(ra1, dec1, radius, ra2, dec2, nside=nside, maxmatch=maxmatch)
#print (matches)

ra1matched  = ra1[ matches['i1'] ]
dec1matched = dec1[ matches['i1'] ]
mag1matched = mag1[ matches['i1'] ]
magger1matched = magger1[ matches['i1'] ]
ra2matched  = ra2[ matches['i2'] ]
dec2matched = dec2[ matches['i2'] ]
kmag2matched = kmag2[ matches['i2'] ]
ekmag2matched = ekmag2[ matches['i2'] ]


cosgamma=[]
for i in range(len(matches)):
    gamma= mt.cos(90-dec1matched[i])*mt.cos(90-dec2matched[i])+mt.sin(90-dec1matched[i])*mt.sin(90-dec2matched[i])*mt.cos(ra1matched[i]-ra2matched[i])
    cosgamma.append(mt.acos(gamma)*3600)



sns.distplot(cosgamma, kde=False)
plt.savefig("Distancia_angular_entre_matches.jpg")
#print(np.max(cosgamma))

result= sns.jointplot(x=kmag2matched, y=mag1matched, kind="reg", truncate=False, xlim=(5, 25), ylim=(5, 25), color="m", height=7)
plt.plot(kmag2matched, mag1matched, ".")
plt.savefig("Distancia_angular_entre_matches1.jpg")

#print (result)

mask = kmag2matched>12.3


kmagFinal = kmag2matched[mask]

mag1Final = mag1matched[mask]

#Hacer error final aplicando la mascara
errmag2Final =  ekmag2matched[mask]

errmag1Final =  magger1matched[mask]

ekmag2Final = ekmag2matched[mask]

p = plt.polyfit(kmagFinal, mag1Final, 1)
print(p[0], p[1])

mag1Finalajuste= p[0]*kmagFinal + p[1]



#x = mag1Finalajuste - kmagFinal

#mag1Finalajuste1= kmagFinal + x[0]
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
out = optimize.leastsq(errfunc, pinit, args=(kmagFinal, mag1Final, errmag1Final), full_output=1)


pfinal = out[0]
covar = out[1]
print (pfinal)
print (covar)
print (errmag1Final)

mag1Finalerr= pfinal[1]*kmagFinal + pfinal[0]

plt.clf()
plt.subplot(1, 1, 1)
#plt.plot(kmag2matched, powerlaw(kmag2matched, amp, index))     # Fit
plt.plot(kmagFinal, mag1Finalajuste) # Azul
plt.plot(kmagFinal, mag1Finalerr) # Naranja
plt.errorbar(kmagFinal, mag1Final, yerr=errmag1Final, xerr=ekmag2Final, fmt='k.')  # Data
plt.legend(('Ajuste lineal', 'Ajuste lineal con error', 'Valores con su error'))
plt.title('Best Fit')
plt.xlabel('KMAG (2MASS)')
plt.ylabel('MAG_AUTO (SHARKS)')
plt.savefig("Ajuste_lineal_con_errores.jpg")















