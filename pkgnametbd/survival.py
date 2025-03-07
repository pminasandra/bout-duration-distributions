# Pranav Minasandra
# pminasandra.github.io
# Dec 29, 2022

"""
This module provides the function compute_behavioural_inertia, which computes
the complement of the hazard function based on a given dataset and state. It
also includes the wrapper function generate_behavioural_inertia_plots(), which
does this analysis on all species, individuals, and behavioural states. 
"""

import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from pkgnametbd import boutparsing
from pkgnametbd import config
from pkgnametbd import classifier_info
from pkgnametbd import fitting
from pkgnametbd import replicates
from pkgnametbd import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

# NOTE: The below function has an argument, hazard_rate, that defaults to False.
# Initially, we thought 1 - Hazard was a better pitch for the paper. However,
# since, the paper has switched to hazard rate in totality. Make sure to always
# turn hazard_rate to True whenever using this function.

def _is_invalid(result):
    if isinstance(result, str):
        if result == "invalid":
            return True

def hazard_function(x, data, hazard_rate=False):
    """
    Returns the hazard function at value x (int) based on available bouts data
    (np.array-like)
    """

    BI_main = data[data > x]
    BI_denom = len(BI_main)
    BI_sub = data[data > (x + 1.0)]
    BI_numer = len(BI_sub)

    if BI_denom == 0:
        return np.nan

    if not hazard_rate:
        return (BI_numer/BI_denom)
    else:
        return 1 - (BI_numer/BI_denom)

def _preprocess_bouts(dataframe, species, state):
    dataframe = fitting.preprocessing_df(dataframe, species)
    bouts = dataframe[dataframe["state"] == state].duration
    if len(bouts) < config.survival_num_points_to_exclude:
        return "invalid" # sort of hacky, unfortunately :(

    return bouts

def _get_unique_times(bouts):
    unique_values = bouts.unique()
    unique_values.sort()
    unique_values = unique_values[:-1] # Excluding the biggest bout because
                                       # it contributes nothing here
    if config.survival_exclude_last_few_points:
        sorted_vals = np.sort(bouts)
        nth_from_last = sorted_vals[-config.survival_num_points_to_exclude]

        unique_values = unique_values[unique_values < nth_from_last]

    return unique_values

def get_BI(bouts, ts=None, hazard_rate=False):
    """
    Get behavioural inertial OR hazard function from bouts data.
    Args:
        bouts (np.array-like of float)
        ts (np.array-like of float): at what values of bout-length to sample
                            the behavioural inertia / hazard function. None
                            means automatic determination of ts.
        hazard_rate (bool): default False, whether to plot hazard rate
                            instead of behavioural inertia.
    Returns:
        np.array (two column): time and inertia values
    """

    if ts is None:
        ts = _get_unique_times(bouts)
    BIs = []

    for t in ts:
        BIs.append(hazard_function(t, bouts, hazard_rate))

    table = np.array([ts, BIs]).T
    return table

def compute_behavioural_inertia(dataframe, species, state, hazard_rate=False):
    """
    Computes behavioural inertia (see docs) for a given dataset.
    Args:
        dataframe (pd.DataFrame): bout information, typically output from
            boutparsing.bouts_data_generator(...).
        species (str): name of species from which the data come.
        state (str): name of the behavioural state.
        hazard_rate (bool): default False, whether to plot hazard rate
                            instead of behavioural inertia.
    Returns:
        np.array (two column): time and inertia values
   """
    bouts = _preprocess_bouts(dataframe, species, state)

    if _is_invalid(bouts):
        return "invalid"
    return get_BI(bouts, hazard_rate=hazard_rate)


def get_bootstrapped_beh_in(bouts, num_replicates, hazard_rate=False):
    """
    Given bouts, bootstraps the data many times to get several estimates of
    the hazard function.
    Args:
        bouts (list-like): all bouts
        num_replicates (int): how many times the data should be bootstrapped
        hazard_rate (bool): default False, whether to compute hazard rate
                            instead of behavioural inertia.
    """

    unique_times = _get_unique_times(bouts)
    data = []
    for bootstrap_bouts in replicates.bootstrapped_data_generator(bouts, num_replicates):
        data.append(get_BI(bootstrap_bouts, ts=unique_times,
                                hazard_rate=hazard_rate)[:,1]) # keep only the
                                                               # hazard rates

    return np.array(data)

def bootstrap_and_analyse(data, species, state, hazard_rate=False):
    """
    Main wrapper function for get_bootstrapped_beh_in(...)
    """
    bouts = _preprocess_bouts(data, species, state)
    ts = _get_unique_times(bouts)
    bootstrap_table = get_bootstrapped_beh_in(bouts,
                        num_replicates=config.NUM_BOOTSTRAP_REPS,
                        hazard_rate=hazard_rate)
    return bootstrap_table


def _get_mean_hazard_rate(bootstrap_haz_table):
    return np.nanmean(bootstrap_haz_table, axis=0)

def _get_95_percent_cis(bootstrap_haz_table):

    num_cols = bootstrap_haz_table.shape[1]
    uppers, lowers = [], []
    for i in range(num_cols):
        hazard_dists = bootstrap_haz_table[:, i]
        hazard_dists = hazard_dists[~np.isnan(hazard_dists)]

        upper_val = np.quantile(hazard_dists, 0.975, method='closest_observation')
        lower_val = np.quantile(hazard_dists, 0.025, method='closest_observation')

        uppers.append(upper_val)
        lowers.append(lower_val)

    return np.array(uppers), np.array(lowers)

def _markovised_seq_parallel_helper(markovised_sequence, count_markov,
                            species_, state, hazard_rate, add_bootstrapping):

    print(f"Working on Markovisation #{count_markov + 1}")
    survival_table_m = compute_behavioural_inertia(
                            markovised_sequence,
                            species_,
                            state,
                            hazard_rate=hazard_rate
                        )
    if _is_invalid(survival_table_m):
        return "invalid"
    if add_bootstrapping:
        bootstrap_table_m = bootstrap_and_analyse(
                                markovised_sequence,
                                species_,
                                state,
                                hazard_rate=hazard_rate
                            )
        upper_lim, lower_lim = _get_95_percent_cis(bootstrap_table_m)
        return survival_table_m, upper_lim, lower_lim

    return survival_table_m

def generate_behavioural_inertia_plots(hazard_rate=False, add_bootstrapping=True,
                                        add_markov=True):
    """
    Generates behavioural inertia vs time plots for all states, individuals,
    species.
    Args:
        add_bootstrapping (bool): default True, whether to estimate and add 95%
                                CIs with bootstrapping.
        add_markov (bool): default True, whether to repeat analysis with
                                Markovised sequences.
        hazard_rate (bool): default False, whether to plot hazard rate instead
                                of behavioural inertia
    """

# Load data
    print("Behavioural inertia plot generation initiated.")
    if add_bootstrapping:
        print(f"For each species-state, we will also bootstrap \
{config.NUM_BOOTSTRAP_REPS} times.")
    if add_markov:
        print(f"For each species-state, we will redo analysis \
with {config.NUM_MARKOVISED_SEQUENCES} Markovised seqeunces.")
    bdg = boutparsing.bouts_data_generator(extract_bouts=False)

    plots = {}
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data_true = databundle["data"]
        data = boutparsing.as_bouts(data_true, species_)
        print(f"Working on {species_} {id_}.")
        if add_markov:
            msg_raw = replicates.load_markovisations_parallel(species_, id_) # use actual
                                                        # length and start
            msg = [boutparsing.as_bouts(seq, species_) for seq in msg_raw]

# Generate empty figure
        if species_ not in plots:
            plots[species_] = {}

        states = fitting.states_summary(data)["states"]

        for state in states:
            if state not in plots[species_]:
                plots[species_][state] = plt.subplots()

# Compute hazard rates
            survival_table = compute_behavioural_inertia(data, species_, state,
                                                        hazard_rate=hazard_rate)

            if _is_invalid(survival_table):
                continue

# Do bootstrapping to get error-bounds
            if add_bootstrapping:
                bootstrap_table = bootstrap_and_analyse(data, species_, state,
                                                    hazard_rate=hazard_rate)
                upper_lim, lower_lim = _get_95_percent_cis(bootstrap_table)

# Make plot
            fig, ax = plots[species_][state]
            ax.plot(survival_table[:,0], survival_table[:,1],
                        color=config.survival_plot_color,
                        linewidth=0.75,
                        alpha=0.4
                    )
            ts = survival_table[:,0]
            if add_bootstrapping:
                ax.fill_between(ts, upper_lim, lower_lim,
                                color=config.survival_plot_color, alpha=0.09)

            ax.set_xscale(config.survival_xscale)
            ax.set_yscale(config.survival_yscale)

# Markovised sequence analysis and plotting
            if add_markov:

                count_markov = range(config.NUM_MARKOVISED_SEQUENCES)
                tgts = zip(msg, count_markov)
                tgts = ((ms, count, species_, state,
                            hazard_rate, add_bootstrapping) for ms, count in tgts)
                
                pool = mp.Pool()
                results_p = pool.starmap(_markovised_seq_parallel_helper, tgts)
                pool.close()
                pool.join()

                results_p = [res for res in results_p if res != "invalid"]
                actual_hazards = [res[0] for res in results_p]

                min_num_bouts = np.inf
                for table_m in actual_hazards:
                    if table_m.shape[0] < min_num_bouts:
                        min_num_bouts = table_m.shape[0]

                ts = actual_hazards[0][:min_num_bouts, 0]
                actual_hazards = [table_m[:min_num_bouts, 1] for table_m in actual_hazards]
                actual_hazards = np.array(actual_hazards)
                mean_haz = _get_mean_hazard_rate(actual_hazards)
                ax.plot(ts, mean_haz,
                            color=config.markovised_plot_color,
                            linewidth=0.75, alpha=0.4)

                if add_bootstrapping:
                    upper_lims = [res[1] for res in results_p]
                    lower_lims = [res[2] for res in results_p]
                    upper_lims = np.array([s[:min_num_bouts] for s in upper_lims])
                    lower_lims = np.array([s[:min_num_bouts] for s in lower_lims])
                    upper_lim = np.nanmean(upper_lims, axis=0)
                    lower_lim = np.nanmean(lower_lims, axis=0)
                    ax.fill_between(ts, upper_lim, lower_lim,
                                color=config.markovised_plot_color, alpha=0.09)



# Saving all plots
    print("Data analysis completed, saving plots.")
    for species in plots:
        epoch = classifier_info.classifiers_info[species].epoch

        for state in plots[species]:
            fig, ax = plots[species][state]
            ax.set_xscale(config.survival_xscale)

            if epoch != 1.0:
                ax.set_xlabel(f"Time ($\\times {epoch}$ seconds)")
            else:
                ax.set_xlabel("Time (seconds)")

#            ax.set_title(f"Species: {species.title()} | State: {state.title()}")
            if hazard_rate:
                ax.set_ylabel("Hazard function")
                utilities.saveimg(fig, f"Hazard-Rate-{species}-{state}")
            else:
                ax.set_ylabel("Behavioural Inertia")
                utilities.saveimg(fig, f"Behavioural-Inertia-{species}-{state}")

    print("Behavioural inertia plot generation finished.")
# Done

if __name__ == "__main__":
    if config.COLLAGE_IMAGES:
        plt.rcParams.update({'font.size': 22})

    generate_behavioural_inertia_plots(hazard_rate=True,
        add_markov=config.ADD_MARKOV,
        add_bootstrapping=config.ADD_BOOTSTRAPPING)
