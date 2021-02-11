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


from astropy import units as u
from astropy.coordinates import SkyCoord
import pylab as pl



class file:
    
    title= "Sharks fits file"
    
    def __init__(self, data):
        
        self.columns =data.columns
        print(self.columns)
        self.pd = data

'''
class read_columns(file):
    
    def columns(self, name):
        
        name=data
'''

class transform_to_lb ():
    def __init__(self,file_object):
        self.file_obj = file_object

    def coordinates(self):
        x=self.file_obj.pd['ALPHA_J2000']*u.degree
        y=self.file_obj.pd['DELTA_J2000']*u.degree
        
        c = SkyCoord(ra=x, dec=y, frame='icrs')
        galactic_coord=c.galactic
        self.file_obj.pd["l"]=galactic_coord.l.deg
        self.file_obj.pd["b"]=galactic_coord.b.deg

        temp_cols = Index(['l', 'b'], dtype='object')

        self.file_obj.columns.append(temp_cols)
        return self.file_obj
    
class plot():
    def __init__(self,file_object):
        self.file_obj = file_object
    
    def get_plot(self, x, y, savename):
        r= sns.jointplot(data=self.file_obj.pd, x=x, y=y, kind='hex')

        pl.savefig(savename)

        '''
        rg= sns.jointplot(data=self.file_obj.pd, x="l", y= "b", kind='hex')
        pl.savefig('test2.png')

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
#        pl.savefig('test.png') 
        '''
       
dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()

f_ = file(df)

tt = transform_to_lb(f_)

new_file_class = tt.coordinates()

p=plot(f_)    
p.get_plot("ALPHA_J2000","DELTA_J2000","plot_radec.png")
p.get_plot("l","b","plot_lb.png")

