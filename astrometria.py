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
import pylab as pl
import pandas as pd


dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()


dat2= Table.read('2mass.fit', format='fits')
df2 = dat.to_pandas()

tips= [df, df2]

ra1 =dat.field('ALPHA_J2000')
dec1 = dat.field('DELTA_J2000')
mag1 = dat.field('MAG_AUTO') 
ra2 = dat2.field('RAJ2000')
dec2 = dat2.field('DEJ2000')
kmag2 = dat2.field('Kmag')


nside=4096 # healpix nside
maxmatch=1 # return closest match
radius= 1/3600.

# ra,dec,radius in degrees
matches = smatch.match(ra1, dec1, radius, ra2, dec2, nside=nside, maxmatch=maxmatch)
#print (matches)

ra1matched  = ra1[ matches['i1'] ]
dec1matched = dec1[ matches['i1'] ]
mag1matched = mag1[ matches['i1'] ]
ra2matched  = ra2[ matches['i2'] ]
dec2matched = dec2[ matches['i2'] ]
kmag2matched = kmag2[ matches['i2'] ]


cosgamma=[]
for i in range(len(matches)):
    gamma= mt.cos(90-dec1matched[i])*mt.cos(90-dec2matched[i])+mt.sin(90-dec1matched[i])*mt.sin(90-dec2matched[i])*mt.cos(ra1matched[i]-ra2matched[i])
    cosgamma.append(mt.acos(gamma)*3600)



#sns.distplot(cosgamma, kde=False)

#print(np.max(cosgamma))

#result= sns.jointplot(x=kmag2matched, y=mag1matched, kind="reg", truncate=False, xlim=(5, 25), ylim=(5, 25), color="m", height=7)
pl.plot(kmag2matched, mag1matched, ".")
#print (result)

mask = kmag2matched>12.3

kmagFinal = kmag2matched[mask]

mag1Final = mag1matched[mask]

p = pl.polyfit(kmagFinal, mag1Final, 1)
print(p[0], p[1])

mag1Finalajuste= p[0]*kmagFinal + p[1]



#x = mag1Finalajuste - kmagFinal

#mag1Finalajuste1= kmagFinal + x[0]

pl.plot(kmagFinal, mag1Finalajuste) #, kmagFinal, mag1Finalajuste1)
pl.plot([10,20],[10,20])

pl.plot(kmag2matched, mag1matched-p[1], ".")
pl.show()
#p2= pl.polyfit(kmagFinal, mag1Finalajuste1, 1)

#print (p2[0], p2[1])


