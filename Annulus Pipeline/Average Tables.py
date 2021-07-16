# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:39:40 2021

@author: jessm

This prints the results of each of the inpainted temporal cube slice analysis 
then averages the results and plots that
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import sys 

from astropy.table import QTable, Table, Column
from astropy import units as u

#I am working with numbers 26, 30, 38, 40, and 44
table26= np.load('table_inpaint_a26.npy')
table30= np.load('table_inpaint_a30.npy')
table38= np.load('table_inpaint_a38.npy')
table40= np.load('table_inpaint_a40.npy')
table44= np.load('table_inpaint_a44.npy')
#print(table26.shape, "\n", table26)

avgtable=(table26+table30)/2

nicetable30=np.vstack((np.round(table30, decimals=2)))
name=Table(nicetable30,  names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
name.pprint_all()

nicetable26=np.vstack((np.round(table26, decimals=2)))
name=Table(nicetable26,  names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
name.pprint_all()

nicetable=np.vstack((np.round(table38, decimals=2)))
name=Table(nicetable,  names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
name.pprint_all()

nicetable=np.vstack((np.round(table40, decimals=2)))
name=Table(nicetable,  names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
name.pprint_all()

nicetable=np.vstack((np.round(table44, decimals=2)))
name=Table(nicetable,  names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
name.pprint_all()

bigavg=(table26+table30+table38+table40+table44)/5

column_len= bigavg.shape[0]

print('\n\n')

   
"""plotting the final averages"""
t = QTable(np.round(bigavg, decimals=2), names=('Pixels', 'Speckles', 'Percent', 'Avg Intensity'))
t.add_column(np.arange(1, 1+column_len), name='Annulus', index=0)
#print(t)
t.pprint(align='^')