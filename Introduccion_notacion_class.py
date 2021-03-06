# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:55:23 2020

@author: lpama
"""
import numpy as np
from astropy import *
from pandas import *
import seaborn as sns
from matplotlib.pyplot import *
from astropy.table import Table

dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()


from astropy import units as u
from astropy.coordinates import SkyCoord
import pylab as pl



class file:
    
    title= "Sharks fits file"
    
    def __init__(self, data):
        
        data=df.columns
    
class read_columns(file):
    
    def columns(self, name):
        
        name=data
        
class transform_to_lb (file):
    
    def coordinates(self, x, y):
        x=dat.field('ALPHA_J2000')
        y=dat.field('DELTA_J2000')
        
        c = SkyCoord(ra=x, dec=y, frame='icrs')
        galactic_coord=c.galactic
        df["l"]=galactic_coord.l.deg
        df["b"]=galactic_coord.b.deg
    
class plot(file):
    
    def get_plot(self, r, rg):
        r= sns.jointplot(data=df, x="ALPHA_J2000", y= "DELTA_J2000", kind='hex')
        rg= sns.jointplot(data=df, x="l", y= "b", kind='hex')
        
        #subplots migration
        f = pl.figure()
        for J in [r, rg]:
                   for A in J.fig.axes:
                                  f._axstack.add(f._make_key(A), A)
        #subplots size adjustment
        f.axes[0].set_position([0.05, 0.05, 0.4,  0.4])
        f.axes[1].set_position([0.05, 0.45, 0.4,  0.05])
        f.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
        f.axes[3].set_position([0.55, 0.05, 0.4,  0.4])
        f.axes[4].set_position([0.55, 0.45, 0.4,  0.05])
        f.axes[5].set_position([0.95, 0.05, 0.05, 0.4])
    
       
       
    
    
    


     
        
        