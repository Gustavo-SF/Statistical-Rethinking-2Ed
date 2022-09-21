import pathlib

import pandas as pd
import numpy as np


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