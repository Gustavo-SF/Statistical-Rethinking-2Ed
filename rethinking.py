import pathlib

import pandas as pd
import numpy as np
import pymc as pm
from scipy import stats


DATA_PATH = pathlib.Path('..') / '..' / 'data'


def standardize(data: np.ndarray):
    return (data - data.mean()) / data.std()


def precis(posterior, var_names):
    """Generate a table that summarizes the distribution of the
    posterior.
    
    Args:
        posterior: The posterior multivariate normal distribution.
        var_names: The names of each variable in the same order as
            the posterior columns.
    """
    post = posterior.rvs(size=10_000)
    data = pd.DataFrame(post, columns=var_names)
    
    # Gather the data
    mean_ = data.mean()
    std_ = data.std()
    quantile055 = data.quantile(0.055)
    quantile945 = data.quantile(0.945)
    
    precis_table = pd.concat([
        mean_, std_, quantile055, quantile945], axis=1)
    
    precis_table.columns = [
        'mean', 'std', '5.5%', '94.5%'
    ]
    return precis_table


def quap(model):
    vars_ = [
        var 
        for val in model.value_vars 
        for var, values in model.rvs_to_values.items() 
        if val == values
    ]
    var_names = [v.name for v in vars_]
    mean_q = pm.find_MAP()
    H = pm.find_hessian(mean_q, vars_)
    cov = np.linalg.inv(H)
    mean = np.concatenate([np.atleast_1d(mean_q[var_]) for var_ in var_names])
    posterior = stats.multivariate_normal(mean=mean, cov=cov)
    return posterior, vars_