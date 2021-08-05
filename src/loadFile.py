import re
import numpy as np
import pandas as pd

def loadFile(path, headers=None):
    '''
    This function will parse a standard HEKA .asc file into a pandas dataframe.

    Arguments: 
    path - a stringIO input of a standard HEKA output .asc file.

    Returns:
    df, dfcache - two copies of the file reformatted into a dataframe.
    '''

    lineIndices = []            
    
    # Splits string at \n and removes trailing spaces  
    with open(path, "r") as f:                        
        rawFile = f.read().strip().split("\n")         

    count=0
    # Finds rows that contain header information to exclude from df                                     
    for line in rawFile:                                  
        if re.search(r"[a-z]+", line) == None:           
            lineIndices.append(count)                     
        count += 1                                    
    
    # Formats headerless file for later df
    processedFile = [rawFile[i].strip().replace(" ", "").split(",") for i in lineIndices]     

    # Use the difference in file size with and without headers to find nSweeps
    nSweeps = int((len(rawFile)-len(processedFile)-1)/2)   

    if headers == None:
         df = pd.DataFrame(data=processedFile)
    else:
        df = pd.DataFrame(columns=headers, data=processedFile)
    df = df.apply(pd.to_numeric)
    df = df.dropna(axis=0)

    # Make new column with sweep identity
    df['sweep'] = np.repeat(np.arange(nSweeps) + 1, len(df)/nSweeps)
    return df.reset_index(drop=True)