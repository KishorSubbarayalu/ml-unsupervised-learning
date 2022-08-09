# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:10:49 2022

@author: Admin
"""

import myUtils as ut
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

#%%

# Get raw data from instagram using a hashtag
url = "https://instagram-data1.p.rapidapi.com/hashtag/feed"

hashtag = input("Enter the instagram hashtag:\n")

payload = {"hashtag":hashtag}

headers = {
	"X-RapidAPI-Key": "XXXX", # Enter API key
	"X-RapidAPI-Host": "YYYY" # Enter API Host
}

raw_data = ut.getData('GET',url,headers=headers,payload=payload)

del url, payload, headers

#%%

# Process the raw data and convert into dataframe

raw_data_cpy = raw_data.copy()

requiredkey = ['collector']

ut.removeKeysInDictionary(raw_data_cpy,requiredkey)

raw_data_cpy = raw_data_cpy.pop('collector')

#%%

# Continue processing the raw data

requiredkey = ['comments','hashtags','likes']

for dictionary in raw_data_cpy:
    ut.removeKeysInDictionary(dictionary,requiredkey)
    
del dictionary, requiredkey
    
#%%

# Create a dataframe

json_string = ut.getJsonDumps(raw_data_cpy)

instapostsdf = ut.loadData(path=json_string,ftype='json')

del json_string, raw_data_cpy

#%%

# Feature engineer additional columns

# Add a new column that has the total number of hastags per post

instapostsdf['totalhashtags'] = instapostsdf.hashtags.apply(
                                                lambda x: len(x))

#%%

# Explore the data and get null percentage

ut.exploreData(instapostsdf)
ut.getNullPercent(instapostsdf)

#%%

# Draw pair wise plot of numeric features

sb.pairplot(instapostsdf.loc[:,instapostsdf.columns != 'hashtags']).fig. \
    suptitle('Pair wise plot', y=1.05)
    
#%%

# Perform standardization to bring the features on same scale

scale = StandardScaler()

scaledfeatures = scale.fit_transform(instapostsdf.loc[:,
                                        instapostsdf.columns != 'hashtags'])

#%%

# Build the model and fit

optics_clustering = OPTICS(min_samples=6,
                           max_eps=2.0).fit(scaledfeatures)

#%%

del scale, scaledfeatures

instapostsdf['labels'] = optics_clustering.labels_

#%%

sb.pairplot(instapostsdf.loc[:,instapostsdf.columns != 'hashtags'],
            hue='labels',
            palette='icefire').fig. \
    suptitle('Pair wise plot with clusters', y=1.05)

#%%

# The label with -1 is outliers, print the count of observations for each 
# cluster

print(instapostsdf['labels'].value_counts())

#%%

# Save the file

instapostsdf.to_pickle('./Data/instapostsdf')

#%%


