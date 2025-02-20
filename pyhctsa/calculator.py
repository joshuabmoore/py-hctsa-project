import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import os


class Calculator:
    """Compute all univariate time series features.
    
    The calculator takes in a univariate time-series dataset of N instances and returns a 
    feature matrix of size N x F where F is the number of features.

    """

    def __init__(self, dataset=None, name=None, configfile=None):
        
        # define a configfile by sb

