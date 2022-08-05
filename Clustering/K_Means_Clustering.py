# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:37:09 2022

@author: Admin
"""

from utils import loadData
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

pwrcons2016df_feat = pwrcons2016df.iloc[:,:4]

#%%

#2. Pair wise plot

sb.pairplot(pwrcons2016df_feat,
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

print(f'Within Cluster Sum of Squares = {kMeans.inertia_}')
print(f'Cluster Centroids = {kMeans.cluster_centers_}')
print(f'Iterations taken for cluster convergence = {kMeans.n_iter_}')

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
    kMeans.fit(pwrcons2016df_feat)
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

pwrcons2016df_feat['Cluster_ELbow_Unscaled'] = kMeans.labels_

#%%

# Pairwise plot with cluster differentiation

sb.pairplot(pwrcons2016df_feat,
            hue='Cluster_ELbow_Unscaled',
            palette='Spectral',
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
                n_clusters = 2,
                n_init = 10,
                max_iter = 100,
                random_state = 42).fit(scaled_features)

pwrcons2016df_feat['Cluster_ELbow_Scaled'] = kMeans.labels_


sb.pairplot(pwrcons2016df_feat.
            loc[:,pwrcons2016df_feat.columns != 'Cluster_ELbow_Unscaled'],
            hue = 'Cluster_ELbow_Scaled',
            palette='rocket',
            corner = False).fig.suptitle(
                'Pair-Wise Plot With Clusters - Scaled Features',
                 y=1.05)

# Using standardization, the optimal number of clusters were 2,

#%%

#6. Finding the optimum cluster

# 6b. Silhoutte Score

silhouette_coeff = []

for k in range(2,11):
    kMeans = KMeans(n_clusters = k,**kmeans_param)
    kMeans.fit(pwrcons2016df_feat.iloc[:,:4])
    score = silhouette_score(pwrcons2016df_feat.iloc[:,:4], kMeans.labels_)
    silhouette_coeff.append(score)

#%%
    
# Plot the silhoutte coefficients

sb.lineplot(x=range(2,11),
            y=silhouette_coeff).set(
                        title='Silhoutte Score - Unscaled Features',
                        xlabel = 'K',
                        ylabel = 'Silhoutte Coefficients')
                                    
# As per the the plot, the optimal cluster identified using silhoutte score 
# is 2 for unscaled feature
                       
#%%


silhouette_coeff_scaled = []

for k in range(2,11):
    kMeans = KMeans(n_clusters = k,**kmeans_param)
    kMeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kMeans.labels_)
    silhouette_coeff_scaled.append(score)

sb.lineplot(x=range(2,11), 
            y=silhouette_coeff_scaled).set(
                              title='Silhoutte Score - Scaled Features',
                              xlabel = 'K',
                              ylabel = 'Silhoutte Coefficients')

#As per the the plot, the optimal cluster is 2 for Scaled feature
                
#%%

#Fit the model with two clusters and visualize the results

kMeans = KMeans(init = 'random',
                n_clusters = 2,
                n_init = 10,
                max_iter = 100,
                random_state = 42).fit(pwrcons2016df_feat)

pwrcons2016df_feat['Cluster_Silhoutte'] = kMeans.labels_

choose_cols = ~pwrcons2016df_feat.columns.isin(['Cluster_ELbow_Unscaled','Cluster_ELbow_Scaled'])
sb.pairplot(pwrcons2016df_feat.loc[:,choose_cols],
            hue = 'Cluster_Silhoutte',
            palette='rocket',
            corner = False).fig.suptitle(
            'Pair-Wise Plot With Clusters - Scaled Features - SC',
             y=1.05)
del choose_cols

#%%


