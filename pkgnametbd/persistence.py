# Pranav Minasandra
# pminasandra.github.io
# Sep 27, 2023

"""
Implements Detrended Fluctuation Analysis and Predictive Information for
behavioural data.
Available functions:
    1. generate_time_series: Generates a time-series of an indicator function
        for the given state.
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
from scipy.stats import linregress

from pkgnametbd import boutparsing
from pkgnametbd import config
from pkgnametbd import classifier_info
from pkgnametbd import fitting
import replicates
from pkgnametbd import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def generate_time_series(dataframe, species, state):
    """
    Generates a time-series of an indicator function for the given state.
    Args:
        dataframe (pd.DataFrame): bout information, typically output from
            boutparsing.bouts_data_generator(...).
        species (str): name of species from which the data come.
        state (str): name of the behavioural state.
    Returns:
        np.array: desired time series
    """

    # first generate a 1, -1 time series
    timeseries = np.zeros(shape=len(dataframe["state"]))
    timeseries[dataframe["state"] == state] = 1
    timeseries[dataframe["state"] != state] = -1

    return timeseries


def alpha_dfa(timeseries, integrate=False):
    """
    Generates the value of alpha for the time-series. (see docs)
    Args:
        timeseries (np.array): from generate_time_series
        integrate (bool): default False, whether to integrate the -1,1 sequence
            of behaviours. (See the paper)
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
        integrate (bool): whether to integrate the time-series prior to
            performing the analysis (default False).
    Returns:
        list: elements of list are tuples of the form
            (species, state, ind_id, alpha_DFA)
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
    """
    Saves results from compute_all_alpha_dfa(...)
    """
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
                # should be just one value by now
                singleton_res = [res for res in res_sub2 if res[2] == id_]
                assert len(singleton_res) == 1
                dict1[state][id_] = singleton_res[0][-1]

        spdata = pd.DataFrame(dict1)
        print("\n", spdata, sep="")
        spdata.to_csv(os.path.join(config.DATA, f"DFA_{species}.csv"),
                        index_label="id")


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
 

    mask =  np.isclose(all_tdiffs, tdiff*epoch) #mask
    return mask


def _get_random_rows(data, num):
    total_rows = data.shape[0]
    inds = np.random.choice(total_rows, size=num, replace=False)

    return data.copy()[inds, :]

def _linear_regression_with_error(xvals, yvals):
    """
    Calculate the y-intercept of a linear regression and its error.

    Parameters:
        xvals (array-like): The independent variable values.
        yvals (array-like): The dependent variable values.

    Returns:
        tuple: y-intercept and its error (intercept, error)
    """
    # Perform linear regression
    result = linregress(xvals, yvals)
    intercept = result.intercept
    intercept_error = result.intercept_stderr

    return intercept, intercept_error

def _bialek_mp_helper(subsize, data):
    data_subset = _get_random_rows(data, subsize)
    xs = data_subset[:,0]
    ys = data_subset[:,1]

    mi = _mutual_information(xs, ys)
    return subsize, mi

def bialek_corrected_mi(x, y):
    """
    For given x and y values, performs a Slonim-Bialek finite size effect correction,
    and provides the appropriate mutual information value.
    Args:
        x (np.array like): a dataset.
        y: similar to x.
    Returns:
        mi_val (float), mi_err (float)
    """

    x = np.array(x.copy())
    y = np.array(y.copy())

    data = np.vstack((x,y))
    data = data.T

    tot_size = len(x)
    sizes = np.logspace(np.log10(tot_size/2), np.log10(0.9*tot_size), 10).astype(int)
    sizes = np.unique(sizes)

    mp_tgts = [(subsize, data)\
                    for subsize in sizes\
                    for i in range(config.NUM_REPS_PER_SUB_SIZE)\
                ]

    pool = mp.Pool()
    linres = pool.starmap(_bialek_mp_helper, mp_tgts)
    pool.close()
    pool.join()
    pool.terminate()

    xvals = np.array([x for x,y in linres])
    yvals = np.array([y for x,y in linres])

    extrap_mi, extrap_mi_err = _linear_regression_with_error(1/xvals, yvals)
    return extrap_mi, extrap_mi_err


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

    if len(array_starts) < 1000:
        return np.nan, np.nan
    mi, error = bialek_corrected_mi(array_starts, array_ends)
    mi = max(mi, 0.0)
    print("Computations finished for delay", T)
    return mi, error


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

    mis = [mi_val for mi_val, mi_err in mi_vals]
    mi_errs = [mi_err for mi_val, mi_err in mi_vals]

    return np.array(mis), np.array(mi_errs)


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

def _save_best_dist_params(funcs, params, r2vals):
    max_r2 = -np.inf
    best_fit = None
    best_params = None

    for candidate, paramset, r2 in zip(funcs, params, r2vals):
        if r2 > max_r2:
            max_r2 = r2
            best_fit = candidate.__name__
            best_params = paramset

    if best_fit == "exponential_fit":
        return {"best_fit":best_fit,
                    "coefficient": best_params[0],
                    "power_exponent": np.nan,
                    "exponential_decay": best_params[1]}

    if best_fit == "powerlaw_fit":
        return {"best_fit":best_fit,
                    "coefficient": best_params[0],
                    "power_exponent": best_params[1],
                    "exponential_decay": np.nan}

    if best_fit == "truncated_powerlaw_fit":
        return {"best_fit":best_fit,
                    "coefficient": best_params[0],
                    "power_exponent": best_params[1],
                    "exponential_decay": best_params[2]}

def complete_MI_analysis(add_markov=True):
    """
    Runs all analyses for MI decay.
    """

# Load data and inititalise
    print("Mutual Information decay analysis initiated.")

    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    timelags = _time_slots_for_sampling(1, 5000, 50)
    timelags = np.unique(timelags)
    # the first value is duplicated b/c log-scaling + rounding, therefore
    # unique()

    print("complete_MI_analysis: will work on the following lags: ", *timelags)
    plots = {}
    r2_results = []
    param_results = []

    saved_res = ["species", "id", "mi_vals", "mi_errs" "mean_mi_markov",
                    "ulim_mi_markov", "llim_mi_markov", "tls_markov"]
    saved_res = pd.DataFrame(columns=saved_res)

    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]
        table_row = {"species": species_, "id": id_}

# Make empty plots
        if species_ not in plots:
            plots[species_] = plt.subplots()
        fig, ax = plots[species_]
        data["datetime"] = pd.to_datetime(data["datetime"])

        print(f"MI decay analysis working on {species_} {id_}.")

# Compute time-lagged MI values
        mi_vals, mi_errs = mutual_information_decay(data, species_, timelags)
        table_row["mi_vals"] = mi_vals
        table_row["mi_errs"] = mi_errs

# Make plots of actual MI decay
        ax.plot(timelags, mi_vals, color="black", linewidth=0.4)
        ax.fill_between(timelags, mi_vals + 2.58*mi_errs,
                            mi_vals - 2.58*mi_errs,
                            color="black",
                            alpha=0.09)
        ax.set_xscale("log")
        ax.set_yscale("log")

# Markovisation analyses
        if add_markov:
            mvals, merrs = [], []
            markovisations = replicates.load_markovisations_parallel(species_,
                                        id_)
            tls_markov = np.arange(1,21).astype(int)
            tls_markov = np.unique(tls_markov)
            j = 1
            for mark in markovisations:
                # Cannot parallelise this bit because
                # mutual_information_decay(...) already uses a mp.Pool()
                print(f"solving Markovisation #{j}")
                j += 1
                mval, merr = mutual_information_decay(mark, species_, tls_markov)
                mvals.append(mval)
                merrs.append(merr)

            mvals = np.array(mvals)
            mean_mi_markov = np.nanmean(mvals, axis=0)
            ulim_mi_markov = np.nanquantile(mvals, 0.975,
                                    method='closest_observation',
                                    axis=0)
            llim_mi_markov = np.nanquantile(mvals, 0.025,
                                    method='closest_observation',
                                    axis=0)

            valid_vals = llim_mi_markov >= 1e-5
            tls_markov_plot = tls_markov[valid_vals]
            ulim_mi_markov = ulim_mi_markov[valid_vals]
            llim_mi_markov = llim_mi_markov[valid_vals]
            mean_mi_markov = mean_mi_markov[valid_vals]

            table_row["mean_mi_markov"] = mean_mi_markov
            table_row["ulim_mi_markov"] = ulim_mi_markov
            table_row["llim_mi_markov"] = llim_mi_markov
            table_row["tls_markov"] = tls_markov_plot

            if saved_res.empty:
                saved_res = pd.DataFrame(table_row)
            else:
                saved_res = pd.concat(saved_res, pd.DataFrame(table_row))

            ax.autoscale(enable=False)
            ax.plot(tls_markov_plot, mean_mi_markov,
                        color=config.markovised_plot_color,
                        alpha=0.4, linewidth=0.4)
            ax.fill_between(tls_markov_plot, ulim_mi_markov, llim_mi_markov,
                        color=config.markovised_plot_color,
                        alpha=0.09)
            ax.autoscale(enable=True)


# Fit candidate functions
        (mexp, lambda_), _ = fit_function(timelags, mi_vals, exponential_fit)
        (malpha, alpha), _ = fit_function(timelags, mi_vals, powerlaw_fit)
        (mtrunc, talpha, tlambda_), _ = fit_function(timelags, mi_vals,
                                                        truncated_powerlaw_fit)

        r2s = _R2_best_fits((exponential_fit, powerlaw_fit,
                                truncated_powerlaw_fit),
                            ((mexp, lambda_),
                            (malpha, alpha),
                            (mtrunc, talpha, tlambda_)), timelags, mi_vals)
        r2s_raw = list(r2s.values())
        r2s["species"] = species_
        r2s["id"] = id_
        r2_results.append(r2s)

        pars = _save_best_dist_params((exponential_fit, powerlaw_fit,
                                        truncated_powerlaw_fit),
                                      ((mexp, lambda_),
                            (malpha, alpha),
                            (mtrunc, talpha, tlambda_)), r2s_raw)

        pars["species"] = species_
        pars["id"] = id_
        param_results.append(pars)


# Make plots
        # Commenting these out because all fit to trunc pow-law anyway
        #ax.plot(timelags, _exp_func(timelags, mexp, lambda_), color="blue", linestyle="dotted")
        #ax.plot(timelags, _pl_func(timelags, malpha, alpha), color="red", linestyle="dotted")
        ax.autoscale(enable=False)
        ax.plot(timelags, _tpl_func(timelags, mtrunc, talpha, tlambda_),
                    color="maroon",
                    linestyle="dotted",
                    linewidth=0.5
                )
        ax.autoscale(enable=True)

    pd.DataFrame(r2_results).to_csv(os.path.join(config.DATA,
                                                    "MI_decay_R2s.csv"
                                                ),
                                            index=False)
    pd.DataFrame(param_results).to_csv(os.path.join(config.DATA,
                                                    "MI_decay_params.csv"
                                                    ),
                                                index=False)
    saved_res.to_pickle(os.path.join(config.DATA, "MI_analyses_raw.pkl"))

    for species_ in plots:
        fig, ax = plots[species_]
        if classifier_info.classifiers_info[species_].epoch != 1.0:
            ax.set_xlabel(f"Time lag ($\\times {classifier_info.classifiers_info[species_].epoch}$ seconds)")
        else:
            ax.set_xlabel("Time lag (seconds)")

        ax.set_ylabel("Time-lagged mutual information")
        utilities.saveimg(plots[species_][0], f"MI_decay_{species_}")
        
if __name__ == "__main__":
    complete_MI_analysis()
    #results = compute_all_alpha_dfa()
    #save_dfa_data(results)
