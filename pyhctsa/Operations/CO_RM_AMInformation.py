import numpy as np
import math 

def CO_RM_AMInformation(y, tau = 1):
    """
    A wrapper for rm_information(), which calculates automutal information

    Inputs:
        y, the input time series
        tau, the time lag at which to calculate automutal information

    :returns estimate of mutual information

    - Wrapper initially developed by Ben D. Fulcher in MATLAB
    - rm_information.py initially developed by Rudy Moddemeijer in MATLAB
    - Translated to python by Tucker Cullen

    """

    if tau >= len(y):

        return

    y1 = y[:-tau]
    y2 = y[tau:]

    out = RM_information(y1,y2)

    return out[0]

def RM_histogram2(*args):
    """
    rm_histogram2() computes the two dimensional frequency histogram of two row vectors x and y

    Takes in either two or three parameters:
        rm_histogram(x, y)
        rm_histogram(x, y, descriptor)

    x, y : the row vectors to be analyzed
    descriptor : the descriptor of the histogram where:

        descriptor = [lowerx, upperx, ncellx, lowery, uppery, ncelly]
            lower? : the lowerbound of the ? dimension of the histogram
            upper? : the upperbound of the dimension of the histogram
            ncell? : the number of cells of the ? dimension of the histogram

    :return: a tuple countaining a) the result (the 2d frequency histogram), b) descriptor (the descriptor used)

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen

    """

    nargin = len(args)

    if nargin < 1:
        print("Usage: result = rm_histogram2(X, Y)")
        print("       result = rm_histogram2(X,Y)")
        print("Where: descriptor = [lowerX, upperX, ncellX; lowerY, upperY, ncellY")

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 3:
        print("Error: too many arguments")
        return

    if nargin == 2:
        minx = np.amin(x)
        maxx = np.amax(x)
        deltax = (maxx - minx) / (lenx - 1)
        ncellx = math.ceil(lenx ** (1 / 3))

        miny = np.amin(y)
        maxy = np.amax(y)
        deltay = (maxy - miny) / (leny - 1)
        ncelly = ncellx
        descriptor = np.array(
            [[minx - deltax / 2, maxx + deltax / 2, ncellx], [miny - deltay / 2, maxy + deltay / 2, ncelly]])
    else:
        descriptor = args[2]

    lowerx = descriptor[0, 0]  # python indexes one less then matlab indexes, since starts at zero
    upperx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowery = descriptor[1, 0]
    uppery = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    # checking descriptor to make sure it is valid, otherwise print an error

    if ncellx < 1:
        print("Error: invalid number of cells in X dimension")

    if ncelly < 1:
        print("Error: invalid number of cells in Y dimension")

    if upperx <= lowerx:
        print("Error: invalid bounds in X dimension")

    if uppery <= lowery:
        print("Error: invalid bounds in Y dimension")

    result = np.zeros([int(ncellx), int(ncelly)],
                      dtype=int)  # should do the same thing as matlab: result(1:ncellx,1:ncelly) = 0;

    xx = np.around((x - lowerx) / (upperx - lowerx) * ncellx + 1 / 2)
    yy = np.around((y - lowery) / (uppery - lowery) * ncelly + 1 / 2)

    xx = xx.astype(int)  # cast all the values in xx and yy to ints for use in indexing, already rounded in previous step
    yy = yy.astype(int)

    for n in range(0, lenx):
        indexx = xx[n]
        indexy = yy[n]

        indexx -= 1  # adjust indices to start at zero, not one like in MATLAB
        indexy -= 1

        if indexx >= 0 and indexx <= ncellx - 1 and indexy >= 0 and indexy <= ncelly - 1:
            result[indexx, indexy] = result[indexx, indexy] + 1

    return result, descriptor

def RM_information(*args):
    """
    rm_information estimates the mutual information of the two stationary signals with
    independent pairs of samples using various approaches:

    takes in between 2 and 5 parameters:
        rm_information(x, y)
        rm_information(x, y, descriptor)
        rm_information(x, y, descriptor, approach)
        rm_information(x, y, descriptor, approach, base)

    :returns estimate, nbias, sigma, descriptor

        estimate : the mututal information estimate
        nbias : n-bias of the estimate
        sigma : the standard error of the estimate
        descriptor : the descriptor of the histogram, see also rm_histogram2

            lowerbound? : lowerbound of the histogram in the ? direction
            upperbound? : upperbound of the histogram in the ? direction
            ncell? : number of cells in the histogram in ? direction

        approach : method used, choose from the following:

            'unbiased'  : the unbiased estimate (default)
            'mmse'      : minimum mean square estimate
            'biased'    : the biased estimate

        base : the base of the logarithm, default e

    MATLAB function and logic by Rudy Moddemeijer
    Translated to python by Tucker Cullen
    """

    nargin = len(args)

    if nargin < 1:
        print("Takes in 2-5 parameters: ")
        print("rm_information(x, y)")
        print("rm_information(x, y, descriptor)")
        print("rm_information(x, y, descriptor, approach)")
        print("rm_information(x, y, descriptor, approach, base)")
        print()

        print("Returns a tuple containing: ")
        print("estimate, nbias, sigma, descriptor")
        return

    # some initial tests on the input arguments

    x = np.array(args[0])  # make sure the imputs are in numpy array form
    y = np.array(args[1])

    xshape = x.shape
    yshape = y.shape

    lenx = xshape[0]  # how many elements are in the row vector
    leny = yshape[0]

    if len(xshape) != 1:  # makes sure x is a row vector
        print("Error: invalid dimension of x")
        return

    if len(yshape) != 1:
        print("Error: invalid dimension of y")
        return

    if lenx != leny:  # makes sure x and y have the same amount of elements
        print("Error: unequal length of x and y")
        return

    if nargin > 5:
        print("Error: too many arguments")
        return

    if nargin < 2:
        print("Error: not enough arguments")
        return

    # setting up variables depending on amount of inputs

    if nargin == 2:
        hist = RM_histogram2(x, y)  # call outside function from rm_histogram2.py
        h = hist[0]
        descriptor = hist[1]

    if nargin >= 3:
        hist = RM_histogram2(x, y, args[2])  # call outside function from rm_histogram2.py, args[2] represents the given descriptor
        h = hist[0]
        descriptor = hist[1]

    if nargin < 4:
        approach = 'unbiased'
    else:
        approach = args[3]

    if nargin < 5:
        base = math.e  # as in e = 2.71828
    else:
        base = args[4]

    lowerboundx = descriptor[0, 0]  #not sure why most of these were included in the matlab script, most of them go unused
    upperboundx = descriptor[0, 1]
    ncellx = descriptor[0, 2]
    lowerboundy = descriptor[1, 0]
    upperboundy = descriptor[1, 1]
    ncelly = descriptor[1, 2]

    estimate = 0
    sigma = 0
    count = 0

    # determine row and column sums

    hy = np.sum(h, 0)
    hx = np.sum(h, 1)

    ncellx = ncellx.astype(int)
    ncelly = ncelly.astype(int)

    for nx in range(0, ncellx):
        for ny in range(0, ncelly):
            if h[nx, ny] != 0:
                logf = math.log(h[nx, ny] / hx[nx] / hy[ny])
            else:
                logf = 0

            count = count + h[nx, ny]
            estimate = estimate + h[nx, ny] * logf
            sigma = sigma + h[nx, ny] * (logf ** 2)

    # biased estimate

    estimate = estimate / count
    sigma = math.sqrt((sigma / count - estimate ** 2) / (count - 1))
    estimate = estimate + math.log(count)
    nbias = (ncellx - 1) * (ncelly - 1) / (2 * count)

    # conversion to unbiased estimate

    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0

        # conversion to minimum mse estimate

    if approach[0] == 'm':
        estimate = estimate - nbias
        nbias = 0
        lamda = (estimate ** 2) / ((estimate ** 2) + (sigma ** 2))
        nbias = (1 - lamda) * estimate
        estimate = lamda * estimate
        sigma = lamda * sigma

        # base transformations

    estimate = estimate / math.log(base)
    nbias = nbias / math.log(base)
    sigma = sigma / math.log(base)

    return estimate, nbias, sigma, descriptor
