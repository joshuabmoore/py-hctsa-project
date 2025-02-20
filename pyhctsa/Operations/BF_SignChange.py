import numpy as np

def BF_SignChange(y, doFind=0):
    """
    Where a data vector changes sign.

    """
    if doFind == 0:
        return (np.multiply(y[1:],y[0:len(y)-1]) < 0)
    indexs = np.where((np.multiply(y[1:],y[0:len(y)-1]) < 0))

    return indexs
