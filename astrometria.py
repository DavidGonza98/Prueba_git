# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:04:55 2021

@author: lpama
"""
from astropy.table import Table
import smatch


dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()

dat2= Table.read('2mass.fit', format='fits')
df2 = dat.to_pandas()



ra1 =dat.field('ALPHA_J2000')
dec1 = dat.field('DELTA_J2000')
ra2 = dat2.field('RAJ2000')
dec2 = dat2.field('DEJ2000')



nside=4096 # healpix nside
maxmatch=1 # return closest match
radius= 1/3600.

# ra,dec,radius in degrees
matches = smatch.match(ra1, dec1, radius, ra2, dec2, nside=nside, maxmatch=maxmatch)