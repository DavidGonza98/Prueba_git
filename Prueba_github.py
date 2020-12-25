# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:32:51 2020

@author: lpama
"""

from astropy import *
from pandas import *
import seaborn as sns
from matplotlib.pyplot import *
from astropy.table import Table
dat = Table.read('Sharks_sgp_e_2_cat_small.fits', format='fits')
df = dat.to_pandas()


#Comment by Aurelio: necesitaras este import de pylab para poder ver la figura
import pylab as pl


#Comment by Aurelio: el problema era que en data, tenias que poner tu DataFrame de entrada, en este caso: df, y de ahi se leen las columnas x,y. 
#Igualmente, kind tiene que venir entre aspas o comillas, pues es un string.
#Lo correcto sera:
r= sns.jointplot(data=df, x="ALPHA_J2000", y= "DELTA_J2000", kind='hex')

#Excercise: busca la documentacion de jointplot para ver como cambiar el nombre de los ejes, y en vez de ser ALPHA_J2000 y DELTA_J2000, que sea RA y DEC. Con tamano de fuente = 16. 
#Una pista, la funcion se llama set_axis_labels

#r.set_axis_labels('RA', 'DEC')

#Comment: estas siguiente linea hace que los margenes de los plots se ajusten a la figura
#pl.tight_layout()

#Comment: este comando te muestra la figura, si en vez de eso, quieres guardarla, en vez de show(), debes usar la funcion savefig(). Busca informacion para ver como indicar el nombre del archivo de salida. Cuando tengas hecha la figura, formato jpg o png, me la puedes enviar por email y continuar con los siguientes ejercicios que te habia indicado.


from astropy import units as u
from astropy.coordinates import SkyCoord
x=dat.field('ALPHA_J2000')
y=dat.field('DELTA_J2000')
#print(x, y)
c = SkyCoord(ra=x, dec=y, frame='icrs')
galactic_coord=c.galactic
print(galactic_coord)
df["l"]=galactic_coord.l.deg
df["b"]=galactic_coord.b.deg

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

#pl.savefig("Imagen apartado 5_b.jpg")

