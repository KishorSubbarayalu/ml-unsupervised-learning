# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 20:15:35 2022

@author: Admin
"""

import myUtils as ut
import pandas as pd
# from sklearn.cluster import AgglomerativeClustering
# import scipy.cluster.hierarchy as sch
# import matplotlib.pyplot as plt
# import seaborn as sb
# import numpy as np

path = './Data'
filename = 'all_bikez_raw.zip'

#%%

files = ut.extractAllFiles(path,filename)

#%%

bikespecdf = ut.loadData(path,files[0],',')

pd.set_option('display.max_columns', 100)

print(bikespecdf.head())

#%%
    
ut.exploreData(bikespecdf)

#%%

# Create a copy of master

bikespeccpydf = bikespecdf.copy()

# Count null values

ut.getNullPercent(bikespeccpydf)

# Drop the columns if the null percentage is above a threshold

bikespeccpydf = ut.removeColumns(df=bikespeccpydf,threshold=40)

#%%

# Clean the dataset having null percentage less than threshold

# Check the null percentage

ut.getNullPercent(bikespeccpydf)

# On dropping the rows having null records would result in only 15% of the
# total records

# Hence followed the below approach
    # 1. Identify the un necessary columns and drop it
    # 2. Check the null percentage
    # 3. Wrangle the data, convert into numeric with single unit
    # 4. Understand the data distribution
    # 5. Impute the missing values accordingly

# Following are the categorical columns, checked the information on each
# feature and came to this conlusion

# Unncessary features -> unnecessaryFeatures
    # ['Model',
    #  'Year',
    #  'Category',
    #  'Rating',
    #  'Engine type',
    #  'Transmission type',
    #  'Color options',
    #  'Bore x stroke',
    #  'Fuel system',
    #  'Cooling system',
    #  'Front suspension',
    #  'Rear suspension',
    #  'Front tire',
    #  'Rear tire',
    #  'Front brakes',
    #  'Rear brakes',
    #  'Starter',
    #  'Compression',
    #  'Gearbox']

#%%

# 1. Identify the un necessary columns and drop it

unnecessaryFeatures = ['Model','Year','Category','Rating','Engine type',
                       'Transmission type','Color options','Bore x stroke',
                       'Fuel system','Cooling system','Front suspension',
                       'Rear suspension','Front tire','Rear tire',
                       'Front brakes','Rear brakes','Starter','Compression',
                       'Gearbox']

bikespeccpydf = ut.removeColumns(df=bikespeccpydf,
                                 unnecessaryFeatures=unnecessaryFeatures)
#%%

# 2. Check the null percentage

ut.getNullPercent(bikespeccpydf)

#%%

# 3. Data Wrangling

# Implementing a logic as this is custom for this data

## This is not the best method, need to improve the speed

def splitSingleUnit(series):
    
    for i in range(len(series)):
        
        if str(series[i]) == 'nan':
            pass
        else:
            series[i] = series[i].split()[0]
            
    return series

bikespeccpydf = bikespeccpydf.apply(lambda x: splitSingleUnit(x))

#%%

bikespeccpydf['Seat height'] = bikespeccpydf['Seat height'] \
                                .str.replace(',', '').astype(float)

#%%                                
        
bikespeccpydf = bikespeccpydf.apply(lambda x: x.astype('float'))

#%%

    





