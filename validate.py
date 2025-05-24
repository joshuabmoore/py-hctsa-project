import numpy as np
import pandas as pd
import yaml
import time
import importlib
from tqdm import tqdm
from loguru import logger
from functools import partial, wraps
from itertools import product
import os

def zscore_decorator(func):
    @wraps(func)
    def wrapper(y, *args, **kwargs):
        y = BF_zscore(y)
        return func(y, *args, **kwargs)
    return wrapper

def range_constructor(loader, node):
    start, end = loader.construct_sequence(node)
    return list(range(start, end+1))
yaml.add_constructor("!range", range_constructor)

def load_yaml(file):
    print(f"Loading configuration file: {file.split('/')[-1]}")
    funcs = {}
    with open(file) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)

    for module_name in yf:
        print(f"\n*** Importing module {module_name} *** \n")
        module = importlib.import_module(module_name)
        for function_name in yf[module_name]:
            # Get the function's configuration dictionary
            function_config = yf[module_name][function_name]
            # If no configs section exists or if it's empty, use a list with single empty dict
            if ('configs' not in function_config or function_config.get('configs') is None or 
                function_config.get('configs') == []):
                configs = [{}]
            else:
                configs = function_config.get('configs', [{}])

            for params in configs:
                # Handle the case where params is None
                if params is None:
                    params = {}
                    
                zscore_first = params.pop("zscore", False)
                param_keys, param_vals = zip(*params.items()) if params else ([], [])
                
                param_combinations = [dict(zip(param_keys, values)) 
                                   for values in product(*[v if isinstance(v, list) 
                                                        else [v] for v in param_vals])]
                
                # If no parameter combinations were generated, add empty dict
                if not param_combinations:
                    param_combinations = [{}]
                
                # create a function for each parameter combination
                for param_set in param_combinations:
                    feature_name = (f"{module_name}_{function_name}_" + 
                                  "_".join(f"{v}" for k, v in param_set.items())
                                  if param_set else f"{module_name}_{function_name}")
                    if not zscore_first:
                        feature_name += "_raw"
                    
                    print(f"Adding operation {feature_name} with params {param_set} "
                          f"(Z-score={zscore_first})")
                    
                    base_func = partial(getattr(module, function_name), **param_set)
                    if zscore_first:
                        base_func = zscore_decorator(base_func)
                        
                    funcs[feature_name] = base_func
                    
    return funcs