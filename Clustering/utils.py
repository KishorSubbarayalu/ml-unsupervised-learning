# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:43:43 2022

@author: Kishor Subbarayalu
"""

from zipfile import ZipFile
import pandas as pd

def extractAllFiles(path: str, filename: str):
    
    filepath = path+'/'+filename
    with ZipFile(filepath, 'r') as zipobj:
        files = zipobj.namelist()
        zipobj.extractall(path)
    
    zipobj.close()
    
    return files
    
def loadData(path: str, filename: str, delimit: str):
    
    filepath = path+'/'+filename
    df = pd.read_table(filepath, delimiter = delimit,
                       low_memory=False)
    
    return df