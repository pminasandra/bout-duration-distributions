# Pranav Minasandra
# pminasandra.github.io
# Sep 27, 2023

"""
Implements Detrended Fluctuation Analysis for behavioral data.
Available functions:
    1. generate_time_series: Generates a time-series of an indicator function for the given state.
    2. alpha_DFA: computes DFA alpha exponent
    3. compute_all_alpha_dfa: performs the analysis for all data
    4. save_data
"""

import multiprocessing as mp
import os.path

import nolds
import numpy as np
import pandas as pd

import boutparsing
import config
import classifier_info
import fitting
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def generate_time_series(dataframe, species, state):
    """
    Generates a time-series of an indicator function for the given state.
    Args:
        dataframe (pd.DataFrame): bout information, typically output from boutparsing.bouts_data_generator(...).
        species (str): name of species from which the data come.
        state (str): name of the behavioural state.
    Returns:
        np.array: desired time series
    """

    # first generate a 1, -1 time series
    xmin = config.xmin
    epoch = classifier_info.classifiers_info[species].epoch

    timeseries = np.zeros(shape=len(dataframe["state"]))
    timeseries[dataframe["state"] == state] = 1
    timeseries[dataframe["state"] != state] = -1

    return timeseries


def alpha_dfa(timeseries, integrate=False):
    """
    Generates the value of alpha for the time-series. (see docs)
    Args:
        timeseries (np.array): from generate_time_series
        integrate (bool): default True, whether to integrate the -1,1 sequence of behaviours. (See the paper)
    """
    if integrate:
        timeseries = timeseries.cumsum()

    return nolds.dfa(timeseries)


def _mp_helper(data, species_, state, id_, main_list, integrate=False):

    print(f"compute_all_alpha_dfa: working on {species_} {id_}, state {state}")
    timeseries = generate_time_series(data, species_, state)
    alpha = alpha_dfa(timeseries, integrate=integrate)

    main_list.append((species_, state, id_, alpha))

    return None


def compute_all_alpha_dfa(integrate=False):
    """
    Performs DFA and stores the exponent for all individuals and states.
    Args:
        integrate (bool): whether to integrate the time-series prior to performing the analysis (default False).
    Returns:
        list: elements of list are tuples of the form (species, state, ind_id, alpha_DFA)
    """

    manager = mp.Manager()
    MAIN_LIST = manager.list()

    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]

        states = data["state"].unique()


        def _helpful_generator():
            for state in states:
                yield (data, species_, state, id_, MAIN_LIST, integrate)

        pool = mp.Pool()
        pool.starmap(_mp_helper, _helpful_generator())
        pool.close()
        pool.join()

    return MAIN_LIST


def save_data(results):

    unique = lambda iter_: list(set(iter_))
    unique_species = unique([x[0] for x in results])

    for species in unique_species:

        dict1 = {}
        res_sub1 = [res for res in results if res[0] == species]

        unique_ids = unique([x[2] for x in res_sub1])
        unique_states = unique([x[1] for x in res_sub1])

        for state in unique_states:
            dict1[state] = {}
            res_sub2 = [res for res in res_sub1 if res[1] == state]

            for id_ in unique_ids:
                singleton_res = [res for res in res_sub2 if res[2] == id_] # should be just one value by now
                assert len(singleton_res) == 1
                dict1[state][id_] = singleton_res[0][-1]

        spdata = pd.DataFrame(dict1)
        print("\n", spdata, sep="")
        spdata.to_csv(os.path.join(config.DATA, f"DFA_{species}.csv"), index_label="id")


if __name__ == "__main__":
    results = compute_all_alpha_dfa()
    save_data(results)
