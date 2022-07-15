import os
import numpy as np
import pandas as pd
import feather
import scipy.integrate as it
from loadFile import *

def v2nm(V, calibration):
    """
    This function will convert the piezoscanner voltage signal into distance.
    A 20x gain is applied due to the presence of a 20x high-voltage amplifier in series with the waveform generator.
    """

    gain = 20
    dist = gain * calibration * V
    return(dist)

def blSubtraction(times, values, window):
    """
    This function will baseline subtract either the photodetector signal or the
    current by subtracting the mean of the designated time window.
    """

    base = np.mean(values[(times >= window[0]) & (times <= window[1])])
    bl_sub = values - base

    return(bl_sub)

def positionCorrection(z, deflection, calibration=15.21):
    """
    This function remvoes the contribution of the cantilever deflection from the position change.
    """

    val = v2nm(z, calibration) - deflection - min(v2nm(z, calibration))

    return(val)

def calcWork(position, force):
    """
    This function will calculate the cumulative integral of force as a function
    of piezoscanner distance traveled (work).

    Work is returned in units of fJ
    """

    work = it.cumtrapz(force,  x=position,
                       initial=0) 
    work = pd.DataFrame(work*1e-3)

    return(work)



def preprocessFile(path, headers=None, window=[50,150], channel = "k2p"):
    print(path)
    dat = loadFile(path, headers=headers)
    sensitivityDat = pd.read_csv(("_").join(path.split("_", -1)[0:-1]) + '_sensitivity.csv', header=None)
    paramDat = pd.read_csv(("_").join(path.split("_", -1)[0:-1]) + '_params.csv', header=0, index_col=0)

    kcant = float(paramDat.loc['kcant', 'val'])
    dkcant = float(paramDat.loc['dkcant', 'val'])


    ## Change scales to ms, pA, and mV for convenience
    time_cols = [col for col in dat if col.startswith('t')]
    
    dat[time_cols] *= 1e3
    dat['i'] *= 1e12
    dat['v'] *= 1e3
    
    meanSensitivity = np.mean(sensitivityDat[0])
    stdSensitivity = np.std(sensitivityDat[0])

    if max(dat.sweep > 1):
        grps = dat.groupby('sweep')
        dat['i_blsub'] = grps.apply(lambda x: blSubtraction(x.ti, x.i, window)).reset_index(drop=True)
        dat['in0_blsub'] = grps.apply(lambda x: blSubtraction(x.tin0, x.in0, window)).reset_index(drop=True)
        dat['deflection'] = dat['in0_blsub'] * meanSensitivity
        dat['force'] = dat['deflection'] * kcant
        dat['rel_error'] = np.sqrt((stdSensitivity / meanSensitivity) ** 2
                            + (dkcant / kcant) ** 2)

        dat['position'] = grps.apply(lambda x: positionCorrection(x.z, x.deflection)).reset_index(drop=True)
        dat['work'] = grps.apply(lambda x: calcWork(x.position, x.force)).reset_index(drop=True)   
    
    else: 
        dat['i_blsub'] = blSubtraction(dat.ti, dat.i, window)
        dat['in0_blsub'] = blSubtraction(dat.tin0, dat.in0, window)
        dat['deflection'] = dat['in0_blsub'] * meanSensitivity
        dat['force'] = dat['deflection'] * kcant
        dat['rel_error'] = np.sqrt((stdSensitivity / meanSensitivity) ** 2
                            + (dkcant / kcant) ** 2)

        dat['position'] = positionCorrection(dat.z, dat.deflection)
        dat['work'] = calcWork(dat.position, dat.force)

    dat.to_feather(os.path.splitext(path)[0] + '_preprocessed.feather')

def preprocessDirectory(folderPath, protocol, headers, window=[50, 150]):
    """
    This function will run the preprocessFile function on all files in a folder.
    """
    path_list = []

    for root, dirs, files in os.walk(folderPath):
        for file in files:
            if file.find(protocol +'.asc') != -1:
                path_list.append(os.path.join(root, file).replace("\\","/"))
    
    for path in path_list:
        preprocessFile(path, headers=headers, window=window)
