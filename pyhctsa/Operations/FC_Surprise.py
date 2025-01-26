import numpy as np 
from Operations.SB_CoarseGrain import SB_CoarseGrain
from PeripheryFunctions.BF_ResetSeed import BF_ResetSeed
import scipy as sc
import math

def FC_Surprise(y, whatPrior='dist', memory=0.2, numGroups=3, coarseGrainMethod='quantile', 
                numIters=500, randomSeed=None):
    """
    FC_Surprise   How surprised you would be of the next data point given recent memory.

    Coarse-grains the time series, turning it into a sequence of symbols of a
    given alphabet size, num_groups, and quantifies measures of surprise of a
    process with local memory of the past memory values of the symbolic string.

    We then consider a memory length, memory, of the time series, and
    use the data in the proceeding memory samples to inform our expectations of
    the following sample.

    The 'information gained', log(1/p), at each sample using expectations
    calculated from the previous memory samples, is estimated.

    Parameters:
    -----------
    y : array_like
        The input time series
    what_prior : str, optional
        The type of information to store in memory:
        'dist': the values of the time series in the previous memory samples,
        'T1': the one-point transition probabilities in the previous memory samples,
        'T2': the two-point transition probabilities in the previous memory samples.
    memory : int or float, optional
        The memory length (either number of samples, or a proportion of the
        time-series length, if between 0 and 1) (default: 0.2)
    num_groups : int, optional
        The number of groups to coarse-grain the time series into (default: 3)
    coarse_grain_method : str, optional
        The coarse-graining, or symbolization method:
        'quantile': an equiprobable alphabet by the value of each time-series datapoint,
        'updown': an equiprobable alphabet by the value of incremental changes in the time-series values,
        'embed2quadrants': 4-letter alphabet of the quadrant each data point resides in a two-dimensional embedding space.
    num_iters : int, optional
        The number of iterations to repeat the procedure for.
    random_seed : int or None, optional
        Seed for the random number generator

    Returns:
    --------
    dict
        Summaries of the series of information gains, including the
        minimum, maximum, mean, median, lower and upper quartiles, and
        standard deviation.
    """

    if (memory > 0) and (memory < 1): #specify memory as a proportion of the time series length
        memory = int(np.round(memory*len(y)))

    # COURSE GRAIN
    yth = SB_CoarseGrain(y, coarseGrainMethod, numGroups) # a coarse-grained time series using the numbers 1:numgroups
    N = int(len(yth))

    #select random samples to test
    np.random.seed(randomSeed) # set seed for reproducibility
    rs = np.random.permutation(int(N-memory)) + memory # can't do beginning of time series, up to memory
    rs = np.sort(rs[0:min(numIters,(len(rs)-1))])
    rs = np.array([rs]) # made into two dimensional array to match matlab and work with testing code directly below

    # COMPUTE EMPIRICAL PROBABILITIES FROM TIME SERIES

    store = np.zeros([numIters, 1])
    for i in range(0, rs.size): # rs.size
        if whatPrior == 'dist':
            # uses the distribution up to memory to inform the next point
            p = np.sum(yth[np.arange(rs[0, i]-memory-1, rs[0, i]-1)] == yth[rs[0, i]-1])/memory # had to be careful with indexing, arange() works like matlab's : operator
            store[i] = p
        elif whatPrior == 'T1':
            # uses one-point correlations in memory to inform the next point
            # estimate transition probabilites from data in memory
            # find where in memory this has been observbed before, and preceded it
            memoryData = yth[rs[0, i] - memory - 1:rs[0, i]-1] # every outputted value should be one less than in matlab
            # previous data observed in memory here
            inmem = np.nonzero(memoryData[0:memoryData.size - 1] == yth[rs[0, i]-2])
            inmem = inmem[0] # nonzero outputs a tuple of two arrays for some reason, the second one of all zeros
            if inmem.size == 0:
                p = 0
            else:
                p = np.mean(memoryData[inmem + 1] == yth[rs[0, i]-1])
            store[i] = p

        elif whatPrior == 'T2':
            # uses two point correlations in memory to inform the next point
            memoryData = yth[rs[0, i] - memory - 1:rs[0, i]-1] # every outputted value should be one less than in matlab
            inmem1 = np.nonzero(memoryData[1:memoryData.size - 1] == yth[rs[0, i]-2])
            inmem1 = inmem1[0]
            inmem2 = np.nonzero(memoryData[inmem1] == yth[rs[0, i]-3])
            inmem2 = inmem2[0]

            if inmem2.size == 0:
                p = 0
            else:
                p = np.sum(memoryData[inmem2+2] == yth[rs[0, i]-1])/len(inmem2)

            store[i] = p

        else:
            print("Error: unknown method: " + whatPrior)
            return
    
    # INFORMATION GAINED FROM NEXT OBSERVATION IS log(1/p) = -log(p)
    store[store == 0] = 1 # so that we set log[0] == 0

    out = {} # dictionary for outputs
    for i in range(0, len(store)):
        if store[i] == 0:
            store[i] = 1

    store = -(np.log(store))
    #minimum amount of information you can gain in this way
    if np.any(store > 0):
        out['min'] = min(store[store > 0]) # find the minimum value in the array, excluding zero
    else:
        out['min'] = np.nan
        
    # Calculate statistics
    out['max'] = np.max(store) # maximum amount of information you cna gain in this way
    out['mean'] = np.mean(store)
    out['sum'] = np.sum(store)
    out['median'] = np.median(store)
    lq = sc.stats.mstats.mquantiles(store, 0.25, alphap=0.5, betap=0.5) # outputs an array of size one
    out['lq'] = lq[0] #convert array to int
    uq = sc.stats.mstats.mquantiles(store, 0.75, alphap=0.5, betap=0.5)
    out['uq'] = uq[0]
    out['std'] = np.std(store, ddof=1)

    # t-statistic to information gain of 1
    if out['std'] == 0:
        out['tstat'] = np.nan
    else:
        out['tstat'] = abs((out['mean']-1)/(out['std']/math.sqrt(numIters)))

    return out 
