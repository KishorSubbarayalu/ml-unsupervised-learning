# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:22:50 2022

@author: Admin
"""

import myUtils as ut
import seaborn as sb
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

path = './Data'

#%%

# API paramteres

url = "https://realty-in-ca1.p.rapidapi.com/properties/list-residential"

payload = {"LatitudeMax":"81.14747595814636",
               "LatitudeMin":"-22.26872153207163",
               "LongitudeMax":"-10.267941690981388",
               "LongitudeMin":"-136.83037765324116",
               "RecordsPerPage":"50",
               "CultureId":"1"}

headers = {
 	"X-RapidAPI-Key": "XXXX", # Enter API key
 	"X-RapidAPI-Host": "YYYY" # Enter API Host
}

# Fetched raw data
data = {}

for i in range(1,6):
    payload["CurrentPage"] = str(i),
    raw_data = ut.getData('GET',url,headers=headers,payload=payload)
    data[i] = raw_data["Results"]
    
del url, payload, headers, i

#%%

# Converted the raw data into list of dictionaries
inp_data = []

for item in list(data.values()):
    inp_data.extend(item)
    
del item, data
    
#%%

# Removed unnecessary keys

requiredkey = ['Land','Property']

for dictionary in inp_data:
    ut.removeKeysInDictionary(dictionary,requiredkey)
    
del dictionary, requiredkey

#%%

# Function to form the dictionary of lists to create dataframe

out_dict = {}

def dict_walk(dictionary):
    for key,value in dictionary.items():
        if isinstance(value,dict):
            dict_walk(value)
        else:
            if key in out_dict:
                if not isinstance(out_dict[key],list):
                    out_dict[key] = [out_dict[key]]
                out_dict[key].append(value)
            else:
                out_dict[key] = value

#%%

# Executed the function dict_walk

for item in inp_data:
    dict_walk(item)
        
del item

#%%

# Removed unnecessary keys

requiredkey = ['AddressText','Price','SizeTotal','Type']

ut.removeKeysInDictionary(out_dict,requiredkey)    

del requiredkey

#%%

# Created a dataframe

residentialdf = ut.loadData(path=out_dict,ftype='dictionary')


#%%

# Created a copy

residentialcpydf  = residentialdf.copy()


#%%

# Each columns required special treatements

# 1. Split the address text to unit_street, province, zipcode

residentialcpydf[['Unit_Street','Location']] = \
                residentialcpydf.AddressText.str.split('|',expand=True)

#%%
                
# check None records or null values

ut.getNullPercent(residentialcpydf)

# As there was null percentage removed the record

residentialcpydf.dropna(axis = 0, inplace = True)

ut.getNullPercent(residentialcpydf)

#%%

residentialcpydf[['City','Province_Zipcode']] = \
                residentialcpydf.Location.str.split(',',expand=True)

#%%
                
# check None records or null values

ut.getNullPercent(residentialcpydf)

# As there was null percentage removed the record

residentialcpydf.dropna(axis = 0, inplace = True)

ut.getNullPercent(residentialcpydf)
                
#%%
                
for idx,item in enumerate(residentialcpydf.Province_Zipcode):
    if item is None:
        residentialcpydf.loc[idx,'Province'] = 'NaN'
        residentialcpydf.loc[idx,'Zipcode'] = 'NaN'
    else:
        if ' ' in item:
            residentialcpydf.loc[idx,['Province','Zipcode']] = \
                            item.split()
        else:
            if item.isalnum():
                residentialcpydf.loc[idx,'Zipcode'] = item
                residentialcpydf.loc[idx,'Province'] = 'NaN'
            else:
                residentialcpydf.loc[idx,'Province'] = item
                residentialcpydf.loc[idx,'Zipcode'] = 'NaN'
            
del idx, item

#%%

# Remove Unccessary columns

residentialcpydf = ut.removeColumns(df=residentialcpydf,
                 unnecessaryFeatures=['AddressText',
                                      'Unit_Street',
                                      'Location',
                                      'City',
                                      'Province_Zipcode',
                                      'Zipcode'])

#%%
                
# check None records or null values

ut.getNullPercent(residentialcpydf)

# As there was null percentage removed the record

residentialcpydf.dropna(axis = 0, inplace = True)

ut.getNullPercent(residentialcpydf)

#%%

# 2. Add hyphen to the feature 'Type'

residentialcpydf.Type = residentialcpydf.Type.apply(
                        lambda x : x.replace(' ','-'))

#%%

# Check the frequency of each values for the formatted features

print(f'Residential Type : \n{residentialcpydf.Type.value_counts()}')
print()
print(f'Province Located : \n{residentialcpydf.Province.value_counts()}')

#%%

# Strip the leading and trailing spaces for the formatted features

residentialcpydf.Province = residentialcpydf.Province.apply(
                        lambda x : x.strip())

residentialcpydf.Type = residentialcpydf.Type.apply(
                        lambda x : x.strip())

#%%

# Check the frequency of each values for the formatted features

print(f'Residential Type : \n{residentialcpydf.Type.value_counts()}')
print()
print(f'Province Located : \n{residentialcpydf.Province.value_counts()}')

#%%

# Save the data for later use

residentialcpydf.reset_index(drop=True, inplace=True)

residentialcpydf.to_pickle(path+'/'+'residential')

#%%

# 3. Clean the price feature

tmp = residentialcpydf.Price.str.split()

tmp = tmp.apply(lambda x : x[0])

tmp = tmp.apply(lambda x : x.replace('$',''))

tmp = tmp.str.split('/')

#%%

for idx,element in enumerate(tmp):
    if len(element) > 1:
        if element[1] == 'sqft': #Converting the price to acre from sqaure feet
            tmp[idx] = round(float(element[0]) * 43560,2)
            
        elif element[1] == 'm2': # Converting the price to acre from m2
            tmp[idx] = round(float(element[0]) * 4047,2)
            
        else:
            tmp[idx] = float(element[0])
            
    else: # Assumed the price in sqft and convertd to per acre
        tmp[idx] = round(float(element[0]) * 43560,2)
        
del idx, element
        
#%%

residentialcpydf.Price = tmp.copy()

residentialcpydf.Price = residentialcpydf.Price.astype('float64')
    
# Check the null percentage

ut.getNullPercent(residentialcpydf)

del tmp

#%%

# 4. Clean the SizeTotal feature 

sizeTotal = residentialcpydf.SizeTotal.copy()


sizeTotal = sizeTotal.str. \
                    replace(r'(Acre|Acres|acres|acre)','ac',regex=True).str. \
                    replace("|",';',regex=False).str.split(';')
                    
sizeTotal = sizeTotal.apply(lambda x : x[0])
sizeTotal = sizeTotal.apply(lambda x : x.strip())

sizeTotal = sizeTotal.str. \
                    replace('Unknown','NaN',regex=False)
                    
sizeTotal = sizeTotal.str. \
                    replace('Bldg=','',regex=False)

sizeTotal = sizeTotal.str. \
                    replace('under 1/2','0.5',regex=False) 

sizeTotal = sizeTotal.str.split(' ')

#%%

def unitConversion(num,unit):
    if unit == 'sqft':
        return round(num/43560,2)
    elif unit == 'FT':
        return round(num/43560,2)
    elif unit == 'hec':
        return round(num*2.471,2)
    elif unit == 'm2':
        return round(num/4047,2)
    else:
        return num

#%%

for idx,item in enumerate(sizeTotal):
    if ('x' in item) & (len(item) > 3):
        sizeTotal[idx] = unitConversion(round(float(item[0])*float(item[2]),2),
                                        item[3])
    elif ('x' in item):
        sizeTotal[idx] = round(float(item[0])*float(item[2]),2)
    elif len(item) == 1:
        sizeTotal[idx] = 'NaN'
    elif len(item) == 2:
        sizeTotal[idx] = unitConversion(float(item[0]),item[1])

del idx, item
        
#%%

residentialcpydf.SizeTotal = sizeTotal.copy()

residentialcpydf.SizeTotal = residentialcpydf.SizeTotal.astype('float64')
    
# Check the null percentage

ut.getNullPercent(residentialcpydf)

# As there was null percentage removed the record

residentialcpydf.dropna(axis = 0, inplace = True)

ut.getNullPercent(residentialcpydf)

del sizeTotal

#%%
                    
# Save the data for later use

residentialcpydf.reset_index(drop=True, inplace=True)

residentialcpydf.to_pickle(path+'/'+'residential')

#%%                 

ut.exploreData(residentialcpydf)

#%%

# Preprocessing

catColumnsPos = [residentialcpydf.columns.get_loc(col) \
                 for col in list(residentialcpydf.select_dtypes('object').columns)]
    
print('Categorical columns           : {}'.format \
      (list(residentialcpydf.select_dtypes('object').columns)))
    
print('Categorical columns position  : {}'.format(catColumnsPos))

# Standardize the numeric columns

scaled = StandardScaler()

residentialcpydf.iloc[:,[0,2]] = scaled.fit_transform \
                                (residentialcpydf.iloc[:,[0,2]])
                                
#%%

dfMatrix = residentialcpydf.to_numpy()
dfMatrix

# Choose optimal K using Elbow method
cost = []
for cluster in range(1, 10):
    try:
        kprototype = KPrototypes(n_jobs = -1,
                                 n_clusters = cluster,
                                 init = 'Huang', random_state = 0)
        
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break
    
del cluster
    
#%%

# Elbow plot
sb.lineplot(x=range(1,8),
            y=cost).set(title='Elbow Plot',
                       xlabel = 'K',
                       ylabel = 'SSE')
                        
# Optimal clusters = 3
del cost
#%%


# build the model
kprototype = KPrototypes(n_jobs = -1,
                         n_clusters = 3,
                         init = 'Huang',
                         random_state = 0)

kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

#%%

# Add cluster labels to the features

residentialcpydf['label'] = kprototype.labels_

del dfMatrix

#%%

# Unscale the feature
residentialcpydf.iloc[:,[0,2]] = scaled.inverse_transform \
                                (residentialcpydf.iloc[:,[0,2]])
                                         
#%%

# Cluster Analysis


sb.pairplot(residentialcpydf,
            hue='label',
            palette='Spectral',
            corner = False).fig.suptitle('Pair-Wise Plot With Clusters',
                                         y=1.05)
                                         
print('\n-----------------------------------')    
print('Cluster 1')
print(residentialcpydf.loc[residentialcpydf['label'] == 0].iloc[:,[1,3]]. \
      value_counts())
print('\n-----------------------------------')    
print('Cluster 2')
print(residentialcpydf.loc[residentialcpydf['label'] == 1].iloc[:,[1,3]]. \
          value_counts())
print('\n-----------------------------------')    
print('Cluster 3')
print(residentialcpydf.loc[residentialcpydf['label'] == 2].iloc[:,[1,3]]. \
      value_counts())
    
#%%


















