import numpy as np
from scipy.signal import argrelextrema

def period(data, dt=1, cutoff=0.97, use_first=False):
    # find period of saw-tooth signal (assuming no noise)
    maxes = np.array(argrelextrema(data, np.greater)[0])
    maxes = maxes[data[maxes]>=cutoff]
    if len(maxes) > 1:
        if use_first:
            p = (maxes[1:]-maxes[:-1])[0]
        else:
            p = np.mean(maxes[1:]-maxes[:-1])*dt
    else: 
        p = np.inf
    return p

def actionTimes(data, dt=1, cutoff=0.97, use_first=False):
    # find period of saw-tooth signal (assuming no noise)
    maxes = np.array(argrelextrema(data, np.greater)[0])
    maxes = maxes[data[maxes]>=cutoff]
    return maxes

def onset(data, dt=1, cutoff=0.99):
    # find onset of saw-tooth signal (assuming no noise)
    maxes = np.array(argrelextrema(data, np.greater)[0])
    maxes = maxes[data[maxes]>=cutoff]
    if len(maxes):
        o = maxes[0]*dt #np.mean(maxes[1:]-maxes[:-1])
    else:
        o = np.inf
    return o

def neighborMean(data):
    # neighbor mean in a complete graph
    n = len(data)
    mat = np.tile(data,n).reshape([n,n]) + np.diag(np.nan*np.zeros(n))
    neighbor_mean = np.nanmean(mat, axis=1)
    return neighbor_mean