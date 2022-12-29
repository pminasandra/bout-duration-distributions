# Pranav Minasandra
# pminasandra.github.io
# Dec 29, 2022

"""
This module provides the function compute_behavioural_inertia, which computes the complement of the hazard function
based on a given dataset. 
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

import config
import classifier_info
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint


def compute_behavioural_inertia(dataframe, species, state):
    """
    Computes behavioural inertia (see docs) for a given dataset.
    Args:
        dataframe (pd.DataFrame): bout information, typically output from boutparsing.bouts_data_generator(...).
        species (str): name of species from which the data come.
        state (str): name of the behavioural state.
    Returns:
        np.array (two column): time and inertia values
   """
    xmin = config.xmin

    dataframe = dataframe[dataframe["duration"] >= config.xmin].copy()
    dataframe = dataframe[dataframe["state"] == state]

    unique_values = dataframe["duration"].unique()
    unique_values.sort()
    unique_values = unique_values[:-1] #Excluding the biggest bout because it contributes nothing here

    ts = []
    BIs = []

    for t in unique_values:
        BI_main = dataframe[dataframe["duration"] > t]
        BI_denom = BI_main["duration"].count()
        BI_sub = dataframe[dataframe["duration"] > (t + 3.0)] 
        BI_numer = BI_sub["duration"].count()

        ts.append(t)
        BIs.append(BI_numer/BI_denom)

    table = np.array([ts, BIs]).T
    return table

if __name__ == "__main__":
    import boutparsing
    mdg = boutparsing.hyena_data_generator()

    for databundle in mdg:
        survival_table = compute_behavioural_inertia(databundle["data"], databundle["species"], "WALK")
        plt.step(survival_table[:,0], survival_table[:,1])
        plt.xscale('log')
        # plt.yscale('log')
        plt.show()
        break
