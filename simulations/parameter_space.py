# Pranav Minasandra
# pminasandra.github.io
# Dec 20, 2022

import os.path

import numpy as np
from scipy.special import ndtri

import config
import utilities

import simulations.sconfig

# The following if block is necessary because scipy.special.ndtri() is a low level
# function several hundred times faster than scipy.stats.norm.ppf()
if simulations.sconfig.FEATURE_DIST_VARIANCE == 1.0:
    inv_normal_cdf = ndtri
else:
    import scipy.stats.norm
    def inv_normal_cdf(y):
        return scipy.stats.norm.ppf(y, scale=simulations.sconfig.FEATURE_DIST_VARIANCE)

def parameter_values(error_start, error_end, error_num):
    error_start_l = np.log(error_start)/np.log(10)
    error_end_l = np.log(error_end)/np.log(10)
    bayes_errors = np.logspace(error_start_l,
                                error_end_l,
                                error_num
                              )


    means_first_state = inv_normal_cdf(bayes_errors/2)
    means_second_state = -1.0 * means_first_state

    for x, y in zip(means_first_state, means_second_state):
        yield x, y
