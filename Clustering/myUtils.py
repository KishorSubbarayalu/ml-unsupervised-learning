# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:43:43 2022

@author: Kishor Subbarayalu
"""

from zipfile import ZipFile
from typing import TypeVar
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

DataFrame = TypeVar('pandas.core.frame.DataFrame')
Series = TypeVar('pandas.core.frame.Series')

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

def drawHistogram(df: DataFrame,feature: Series,rows: int,cols: int,pos: int):
    plt.subplot(rows,cols,pos)
    plt.title(f'Data distribution of {feature} : Histogram')
    plt.ylabel('Frequency')
    sb.histplot(df[feature],bins=6,kde=True)
    plt.show()
    
def drawBoxplot(df: DataFrame,feature: Series,rows: int,cols: int,pos: int):
    plt.subplot(rows,cols,pos)
    plt.title(f'Data distribution of {feature} : Box plot')
    plt.ylabel('Frequency')
    sb.boxplot(df[feature])
    plt.show()
