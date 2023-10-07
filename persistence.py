# Pranav Minasandra
# pminasandra.github.io
# Sep 27, 2023

"""
Implements Detrended Fluctuation Analysis and Predictive Information for behavioral data.
Available functions:
    1. generate_time_series: Generates a time-series of an indicator function for the given state.
    2. alpha_DFA: computes DFA alpha exponent
    3. compute_all_alpha_dfa: performs the analysis for all data
    4. save_data
"""

import multiprocessing as mp
import os.path

import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, r2_score
import scipy.optimize

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


def save_dfa_data(results):

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


def _time_slots_for_sampling(Tmin, Tmax, num):
    return np.logspace(np.log10(Tmin), np.log10(Tmax), num).astype(int)


def _mutual_information(vals1, vals2):
    mi = adjusted_mutual_info_score(vals1, vals2)
    return mi


def _paired_past_future_indices(array, tdiff):
    num_pairs = len(array) - tdiff
    indices_start = np.arange(num_pairs)
    indices_end = indices_start + tdiff

    return indices_start, indices_end


def _validated_paired_indices(dt_col, tdiff, indices_start, indices_end, epoch):
    s_ends = dt_col[indices_end]
    s_starts = dt_col[indices_start]

    s_ends = s_ends.to_numpy()
    s_starts = s_starts.to_numpy()

    all_tdiffs = s_ends - s_starts
    all_tdiffs *= 1e-9#because numpy measures time in ns for some reason
    all_tdiffs = all_tdiffs.astype(float)
# it sucks to work so much against numpy and pandas in a straightforward implementation of something
    

    mask =  all_tdiffs == tdiff*epoch #mask
    return mask


def MI_t(array, dt_col, T, epoch):
    """
    For a given time-series, quantifies the predictability of the animal's state
    at time t+T given we know the state at time t.
    Args:
        array (pd.Series): behavioral sequence
        dt_col (pd.Series): datetimes
        T (int): lag period, number of epochs
        epoch (float): epoch duration
    Return:
        Mutual information at T delay
    """

    t_starts, t_ends = _paired_past_future_indices(array, T)
    dt_mask = _validated_paired_indices(dt_col, T, t_starts, t_ends, epoch)


    t_starts = t_starts[dt_mask]
    t_ends = t_ends[dt_mask]


    array_starts = array[t_starts]
    array_ends = array[t_ends]

    if len(array_starts) <= 1000:
        return np.nan
    return _mutual_information(array_starts, array_ends)


def mutual_information_decay(df, species, timelags):
    """
    Implements the above analyses for a range of time-lags for an individual.
    Args:
        df (pd.DataFrame): behavioral sequence information, typically from boutparsing.bouts_data_generator(...)
        species (str): name of the species from which the df comes
        timelags (array-like of floats): typically output from _time_slots_for_sampling(...)
    """

    dt_col = df["datetime"]
    sequence = df["state"]

    epoch = classifier_info.classifiers_info[species].epoch

    mi_vals = []

    for tau in timelags:
        mi_vals.append(MI_t(sequence, dt_col, tau, epoch))

    return mi_vals


def _exp_func(x, m, lambda_):
    return m*np.exp(-lambda_*x)

def _pl_func(x, m, alpha):
    return m*x**(-alpha)

def _tpl_func(x, m, alpha, lambda_):
    return m * x**-alpha * np.exp(-lambda_*x)

def exponential_fit(x, *params):
    return _exp_func(x, *params)

def powerlaw_fit(x, *params):
    return _pl_func(x, *params)

def truncated_powerlaw_fit(x, *params):
    return _tpl_func(x, *params)

def fit_function(x, y, func):
    if func.__name__ in ["exponential_fit", "powerlaw_fit"]:
        p0 = (1,1)
    elif func.__name__ == "truncated_powerlaw_fit":
        p0 = (1,1,1)
    params, covar = scipy.optimize.curve_fit(func, x, y, p0=p0)
    return params, covar

def _R2_best_fits(funcs, params, xvals_actual, yvals_actual):
    assert len(funcs) == len(params)
    assert len(funcs) == 3

    r2_vals = {}

    for f, ps in zip(funcs, params):
        fname = f.__name__
        yvals_pred = f(xvals_actual, *ps)
        r2 = r2_score(yvals_actual, yvals_pred)
        r2_vals[fname] = r2
    return r2_vals

def complete_MI_analysis():
    """
    Runs all analyses for MI decay.
    """

    print("Mutual Information decay analysis initiated.")

    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    timelags = _time_slots_for_sampling(1, 5000, 10)
    timelags = np.unique(timelags) #the first value is duplicated b/c log-scaling + rounding

    fig, ax = plt.subplots()
    print("complete_MI_analysis: will work on the following lags: ", *timelags)
    plots = {}
    r2_results = []
    i = 0
    for databundle in bdg:
        if i >= 5:
            break
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]

        data["datetime"] = pd.to_datetime(data["datetime"])

        print(f"MI decay analysis working on {species_} {id_}.")

        mi_vals = mutual_information_decay(data, species_, timelags)
        ax.plot(timelags, mi_vals, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")

        (mexp, lambda_), _ = fit_function(timelags, mi_vals, exponential_fit)
        (malpha, alpha), _ = fit_function(timelags, mi_vals, powerlaw_fit)
        (mtrunc, talpha, tlambda_), _ = fit_function(timelags, mi_vals, truncated_powerlaw_fit)

        r2s = _R2_best_fits((exponential_fit, powerlaw_fit, truncated_powerlaw_fit),
                            ((mexp, lambda_),
                            (malpha, alpha),
                            (mtrunc, talpha, tlambda_)), timelags, mi_vals)
        r2s["species"] = species_
        r2s["id"] = id_
        r2_results.append(r2s)

        if i == 0:
            print(f"Exponential fit: {mexp:.2f} * e**(-{lambda_:.2f}x)")
            print(f"Powerlaw fit: {malpha:.2f} * x**-{alpha:.2f}")
            print(f"Powerlaw fit: {mtrunc} * x**-{talpha:.2f} * e**-{tlambda_:.4f}x")

            ax.plot(timelags, _exp_func(timelags, mexp, lambda_), color="blue", linestyle="dotted")
            ax.plot(timelags, _pl_func(timelags, malpha, alpha), color="red", linestyle="dotted")
            ax.plot(timelags, _tpl_func(timelags, mtrunc, talpha, tlambda_), color="maroon", linestyle="dotted")

        i += 1

    print(pd.DataFrame(r2_results))
    plt.show()
        

    
if __name__ == "__main__":
#    results = compute_all_alpha_dfa()
#    save_data(results)
    complete_MI_analysis()
