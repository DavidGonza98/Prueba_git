# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:32:51 2020

@author: lpama
"""

from astropy import *
from pandas import *
import seaborn as sns

from astropy.table import Table
dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()
#PRINT(type(df))

#Comment by Aurelio: necesitaras este import de pylab para poder ver la figura
import pylab as pl
"""
from astropy.io import fits
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

archivo_fits = fits.open('Sharks_sgp_e_2_cat_small.fits') #open file
"""
#Comment by Aurelio: el problema era que en data, tenias que poner tu DataFrame de entrada, en este caso: df, y de ahi se leen las columnas x,y. 
#Igualmente, kind tiene que venir entre aspas o comillas, pues es un string.
#Lo correcto sera:
plot = sns.jointplot(data=df, x="ALPHA_J2000", y= "DELTA_J2000", kind='hex')

#Excercise: busca la documentacion de jointplot para ver como cambiar el nombre de los ejes, y en vez de ser ALPHA_J2000 y DELTA_J2000, que sea RA y DEC. Con tamano de fuente = 16. 
#Una pista, la funcion se llama set_axis_labels

#Comment: estas siguiente linea hace que los margenes de los plots se ajusten a la figura
pl.tight_layout()

#Comment: este comando te muestra la figura, si en vez de eso, quieres guardarla, en vez de show(), debes usar la funcion savefig(). Busca informacion para ver como indicar el nombre del archivo de salida. Cuando tengas hecha la figura, formato jpg o png, me la puedes enviar por email y continuar con los siguientes ejercicios que te habia indicado.
pl.show()
