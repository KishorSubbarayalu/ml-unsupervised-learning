# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:43:43 2022

@author: Kishor Subbarayalu
"""

from zipfile import ZipFile
from typing import TypeVar
import pandas as pd

DataFrame = TypeVar('pandas.core.frame.DataFrame')

#----------------------------------------------------------------------------#

def extractAllFiles(path: str, filename: str):
    
    filepath = path+'/'+filename
    with ZipFile(filepath, 'r') as zipobj:
        files = zipobj.namelist()
        zipobj.extractall(path)
    
    zipobj.close()
    
    return files
    
def loadData(path: str, filename: str, delimit: str, ftype='txt'):
    
    filepath = path+'/'+filename
    if ftype == 'txt':
        df = pd.read_table(filepath, delimiter = delimit,
                           low_memory=False)
    elif ftype == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        df = pd.read_csv(filepath, delimiter = delimit,
                           low_memory=False)    
    return df

def exploreData(df: DataFrame):
    print('----------------------------------------\n')
    print(f'Number of records:{df.shape[0]}\nNumber of features:{df.shape[1]}')
    print('\n----------------------------------------\n')
    print(f'Feature Information:\n{df.info()}')
    print('\n----------------------------------------\n')
    print(f"Feature Description:\n{df.describe(include='O')}")
    
def getNullPercent(df: DataFrame):
    print('----------------------------------------\n')
    print('The Null Percentage of all the features in the dataset')
    print('\n----------------------------------------\n')
    print((df.isna().sum().sort_values(ascending=False)/df.shape[0]) * 100)
    
def removeColumns(**kwargs):
    df = kwargs['df']
        
    if 'unnecessaryFeatures' in kwargs.keys():
        dropcols = kwargs['unnecessaryFeatures']
    elif 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
        tmp = (df.isna().sum().sort_values(ascending=False)/df.shape[0]) * 100
        dropcols = tmp[tmp>threshold].index.to_list()
    else:
        pass
    
    df = df.drop(dropcols,axis=1)
    
    return df
    
    