# Pranav Minasandra
# pminasandra.github.io
# Dec 26, 2022

"""
Provides methods for fitting data to powerlaw and other distributions.
Mostly uses the module powerlaw.
See: Alstott J, Bullmore E, Plenz D (2014) powerlaw: A Python Package for Analysis
of Heavy-Tailed Distributions. PLoS ONE 9(1): e85777
for more details
"""

import os.path

import numpy as np
import pandas as pd
import powerlaw as pl

import config
import boutparsing
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def states_summary(dataframe):
    """
    Finds all the states and provides a basic quantitative summary of these states.
    Args:
        dataframe (pandas.DataFrame): typically yielded by a boutparsing.bouts_data_generator()
    Returns:
        dict, where
            dict["states"]: list of all states whose bouts are in states summary
            dict["proportions"]: proportion of time the individual was in each state
    """

    states = list(dataframe["state"].unique())
    states.sort()
    total_time = sum(dataframe["duration"])
    results = {"states": states, "proportions": []}
    for state in states:
        state_bouts = dataframe[dataframe["state"] == state]
        state_prop = sum(state_bouts["duration"])/total_time
        results["proportions"].append(state_prop)

    return results


def fits_to_all_states(dataframe, *args, **kwargs):
    """
    Performs powerlaw.Fit for all states separately
    Args:
        dataframe (pandas.DataFrame): typically yielded by a boutparsing.bouts_data_generator()
        args, kwargs: passed on to powerlaw.Fit(...); 
            (see Alstott J, Bullmore E, Plenz D (2014) powerlaw:
            A Python Package for Analysis of Heavy-Tailed Distributions. 
            PLoS ONE 9(1): e85777)
    Returns:
        dict, where for each state,
            dict[state]: powerlaw.Fit object.
    """

    summary = states_summary(dataframe)
    states = summary["states"]

    fitted_distributions = {}

    for state in states:
        state_bouts = dataframe[dataframe["state"] == state]
        durations = state_bouts["duration"]

        fit = pl.Fit(durations, *args, discrete=True, **kwargs)

        fitted_distributions[state] = fit

    return fitted_distributions


def aic(distribution, data):
    if type(distribution).__name__ in ['Lognormal', 'Truncated_Power_Law']:
        params = 2
    elif type(distribution).__name__ in ['Exponential', 'Power_Law']:
        params = 1
    else:
        raise ValueError(f"fitting.aic() has not been programmed for distribution: {distribution}")

    return 2*(params - sum(np.array(distribution.loglikelihoods(data))))

def compare_candidate_distributions(fit, data):
    """
    Computes \delta-AICs for all candidate distributions.
    Args:
        fit (powerlaw.Fit): a fit of bout durations.
        data (list like): the data which was used to fit this distribution.
        args, kwargs: passed on to powerlaw.Fit()
    Returns:
        list: names (str) of distributions
        list: containing \delta-AIC values
    """

    candidates = config.all_distributions(fit)
    AICs = {}
    dAICs
    min_AIC = np.inf
    for candidate_name in candidates:
        candidate = candidates[candidate_name]
        AIC = aic(candidate, data)
        if AIC < min_AIC:
            min_AIC = AIC
        AICs[candidate_name] = AIC

    for candidate_name in candidates:
        AICs[candidate_name] -= min_AIC
        dAICs[candidate_name] =[AICs[candidate_name]] 

    return pd.DataFrame(dAICs)

hdg = boutparsing.hyena_data_generator()
for databundle in hdg:
    print(databundle["id"])
    data = databundle["data"]
    data["duration"] /= 3.0

    fits = fits_to_all_states(data, xmin=1.0, verbose=False)
    states = states_summary(data)["states"]

    for state in states:
        print("\t", state, "\n", compare_candidate_distributions(fits[state], data["duration"]), end="\n\n")

