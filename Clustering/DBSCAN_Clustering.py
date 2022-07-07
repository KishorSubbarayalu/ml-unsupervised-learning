# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 00:02:01 2022

@author: Admin
"""

from utils import extractAllFiles,loadData
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sb
import numpy as np
import pickle

path = './Data'
filename = 'household_power_consumption.zip'

#%%

files = extractAllFiles(path,filename)

#%%

pwrconsdf = loadData(path,files[0],';')

pd.set_option('display.max_columns', 10)

print(pwrconsdf.head())

# %%

# Know the data

print(pwrconsdf.info())
print(pwrconsdf.describe())

#%%

# for learning sake, filter only 2016 records
pwrconsdf['Year'] = pwrconsdf['Date'].apply(lambda x : x[-4:])

pwrconsdf2016 = pwrconsdf.loc[pwrconsdf['Year'] == '2006',]
# %%

print(pwrconsdf2016.info())
print(pwrconsdf2016.describe())

# Looks like all the columns has no expected data type
#%%

# Let's EDA and wrangle the data wherever necessary

# 1. find null values

print(pwrconsdf2016.isna().sum())

feature_cols = ['Global_active_power', 'Global_reactive_power',
                    'Voltage', 'Global_intensity']

# There are no null values in the feature columns

#%%

# 2. convert the type of feature columns to suitable datatype

for cols in feature_cols:
    pwrconsdf2016[cols] = pd.to_numeric(pwrconsdf2016[cols],
                                        errors = 'coerce')

print('\n\n After type conversion {}'.format(pwrconsdf2016.isna().sum()))

#%%

# 3. Drop the null records
pwrconsdf2016.dropna(inplace = True)
print('\n\n After dropping NA records {}'.format(pwrconsdf2016.isna().sum()))

#%% 

# 4. extract feature columns

pwrconsdf2016X = pwrconsdf2016.loc[:,feature_cols]

# %%

# 5. Do pair wise plotting as it has 4 features (multi dimensional)

sb.pairplot(pwrconsdf2016X, corner = False)

#%%

# DBSCAN model instantiation

dbscan = DBSCAN(eps=1,min_samples=3)

#%%

# Model Fit

model = dbscan.fit(pwrconsdf2016X)

core_pt_mask = np.zeros_like(model.labels_, dtype=bool)

core_pt_mask[model.core_sample_indices_] = True
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#%%

pwrconsdf2016X['Cluster'] = labels

sb.pairplot(pwrconsdf2016X[core_pt_mask], hue = 'Cluster', corner = False)

#%%

# The model may not be worked well with high dimensionality
# Let's save the features used and try different clustering technique

pwrconsdf2016X.to_pickle(path+'/'+'pwrconsdf2016X')















