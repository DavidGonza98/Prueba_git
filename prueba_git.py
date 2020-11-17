# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 18:00:29 2020

@author: lpama
"""

from astropy import *
from pandas import *
import seaborn as sns

from astropy.table import Table
dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()
print(type(df))

"""
from astropy.io import fits
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

archivo_fits = fits.open('Sharks_sgp_e_2_cat_small.fits') #open file
"""
sns.jointplot(data=dat, x="ALPHA_J2000", y= "DELTA_J2000", kind=hex)
