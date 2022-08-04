# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:37:09 2022

@author: Admin
"""

from utils import loadData
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sb

path = './Data'
filename = 'pwrconsdf2016X'

#%%

# 1. Read the transformed pickle file
pwrcons2016df = loadData(path = path,
                     filename = filename,
                     ftype = 'pickle',
                     delimit = 'NA')
#%%

#2. Pair wise plot

sb.pairplot(pwrcons2016df.iloc[:,:4],
            corner = False).fig.suptitle('Pair-Wise Plot With No Clusters',
                                         y=1.05)

#%%

#3. Build the model

kMeans = KMeans(init = 'random',
                n_clusters = 3,
                n_init = 10,
                max_iter = 100,
                random_state = 42)

#%%

#4. Fit the model

kMeans.fit(pwrcons2016df.iloc[:,:4])

#5. K-Means Information

print(kMeans.inertia_)
print(kMeans.cluster_centers_)
print(kMeans.n_iter_)

#%%

#6. Finding the optimum cluster

# 6a. Elbow method

kmeans_param = {"init": "random",
                 "n_init": 10,
                 "max_iter": 100,
                 "random_state": 42,
                 }

sse = []

for k in range(1,11):
    kMeans = KMeans(n_clusters = k,**kmeans_param)
    kMeans.fit(pwrcons2016df.iloc[:,:4])
    sse.append(kMeans.inertia_)
    
    
#%%

# Plot the sum of squared errors

sb.lineplot(x=range(1,11),
            y=sse).set(title='Elbow Plot - Unscaled Features',
                       xlabel = 'K',
                       ylabel = 'SSE')

# Looks like the optimum number of cluster is 3 as per elbow plot

kMeans = KMeans(init = 'random',
                n_clusters = 3,
                n_init = 10,
                max_iter = 100,
                random_state = 42).fit(pwrcons2016df.iloc[:,:4])

#%%

# Append the labels to dataframe

# Append the cluster labels to dataframe

pwrcons2016df_feat = pwrcons2016df.iloc[:,:4]

pwrcons2016df_feat['cluster_kmeans'] = kMeans.labels_

#%%

# Pairwise plot with cluster differentiation

sb.pairplot(pwrcons2016df_feat,
            hue='cluster_kmeans',
            corner = False).fig.suptitle('Pair-Wise Plot With Clusters',
                                         y=1.05)

#%%

# Repleat the steps after performing standardization

scalar = StandardScaler()
scaled_features = scalar.fit_transform(pwrcons2016df_feat.iloc[:,0:4])

sse_scaled = []

for k in range(1,11):
    kMeans = KMeans(n_clusters = k,**kmeans_param)
    kMeans.fit(scaled_features)
    sse_scaled.append(kMeans.inertia_)

sb.lineplot(x=range(1,11), 
            y=sse_scaled).set(title='Elbow Plot - Scaled Features',
                              xlabel = 'K',
                              ylabel = 'SSE')

#%%
kMeans = KMeans(init = 'random',
                n_clusters = 4,
                n_init = 10,
                max_iter = 100,
                random_state = 42).fit(scaled_features)

pwrcons2016df_feat['ScaledClusterLabel'] = kMeans.labels_


sb.pairplot(pwrcons2016df_feat.loc[:,pwrcons2016df_feat.columns != 'cluster_kmeans'],
            hue = 'ScaledClusterLabel',
            corner = False).fig.suptitle(
                'Pair-Wise Plot With Clusters - Scaled Features',
                 y=1.05)

# Although using standardization, the optimal number of clusters were 4,
# the clusters formed nicely when k = 3

#%%



