# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 20:15:35 2022

@author: Admin
"""

import myUtils as ut
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


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

for idx,feature in enumerate(bikespeccpydf.columns):
    ut.drawHistogram(bikespeccpydf,feature,5,1,idx+1)
    ut.drawBoxplot(bikespeccpydf,feature,5,1,idx+1)
    
del idx, feature

#%%

sb.pairplot(bikespeccpydf).fig.suptitle('Pair-Wise Plot With No Clusters',
                             y=1.05)

#%%

# 5. Imputation

# From the the univariate and pair-wise plotting it is evident that there are
# many anomalies in the data.
# We shall impute the missing values, based on the data distribution

# 1. Displacement -> Right skewed data, mean>median>mode
#     As mean is greater, replace the NaN with median
# 2. Power ->  Right skewed data, mean>median>mode
#     As mean is greater, replace the NaN with median
# 3. Fuel Capacity -> The distribution is multimedian
#     Replace the NaN with mean/median
# 4. Wheelbase -> Normal distribution
#     Replace the NaN with mean
# 5. Seat height -> The outliers are large, could be because of the incorrect 
#    measurements
#     Impute the incorrect measurements, and the NaN with median

imputeDict = {'Displacement':np.nanmedian(bikespeccpydf['Displacement']),
              'Power':np.nanmedian(bikespeccpydf['Power']),
              'Fuel capacity':np.nanmedian(bikespeccpydf['Fuel capacity']),
              'Wheelbase':np.nanmean(bikespeccpydf['Wheelbase']),
              'Seat height':np.nanmedian(bikespeccpydf['Seat height'])}

#%%
                                    

# Treat on NaN Values

# just copying to see the difference

bikespeccpydf = bikespeccpydf.fillna(imputeDict)

# Treat incorrect measurements on seat height

bikespeccpydf.loc[bikespeccpydf['Seat height'] > 2000,
                  'Seat height'] = imputeDict['Seat height']

#%%

# Null percent
ut.getNullPercent(bikespeccpydf)

# Pair wise plot
sb.pairplot(bikespeccpydf).fig.suptitle('Pair-Wise Plot after imputation',
                             y=1.05)

#%%

# Save a copy on the cleaned data
bikespeccpydf.to_pickle(path+'/bikespeccleaneddf')

#%%

# Build the dendogram to identify the number of clusters

dendrogram = sch.dendrogram(sch.linkage
                            (np.array(np.array(bikespeccpydf.values)),
                                        method='ward'))

#%%

# Build and fit the model

# From the dendogram, the optimal number of clusters would be 4

hierarchical_cluster = AgglomerativeClustering(n_clusters=4,
                                               affinity='euclidean',
                                               linkage='ward')
hierarchical_cluster.fit(bikespeccpydf)

bikespeccpydf['Cluster label'] = hierarchical_cluster.labels_

#%%

sb.pairplot(bikespeccpydf,
            hue = 'Cluster label',
            palette='Spectral').fig.suptitle('Pair-Wise Plot With Clusters',
                             y=1.05)

#%%






