# Pranav Minasandra
# pminasandra.github.io
# Dec 26, 2022

"""
Provides methods for fitting data to powerlaw and other distributions.
Mostly uses the module powerlaw.
See: Alstott J, Bullmore E, Plenz D (2014) powerlaw: A Python Package for
Analysis of Heavy-Tailed Distributions. PLoS ONE 9(1): e85777 for more details
"""

import os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw as pl

from pkgnametbd import config
from pkgnametbd import classifier_info
from pkgnametbd import boutparsing
from pkgnametbd import utilities
import replicates

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

insufficient_data_flag = config.insufficient_data_flag

def preprocessing_df(dataframe, species):
    """
    Converts bouts to epoch units.
    Ensures that the bouts of 1 epoch are removed, and any other preprocessing
    decided later.
    """

    df = dataframe.copy()
    df["duration"] /= classifier_info.classifiers_info[species].epoch
    df = df[df["duration"] >= config.xmin]

    return df


def states_summary(dataframe):
    """
    Finds all the states and provides a basic quantitative summary of these
    states.
    Args:
        dataframe (pandas.DataFrame): typically yielded by a
        boutparsing.bouts_data_generator()
    Returns:
        dict, where
            dict["states"]: list of all states whose bouts are in states summary
            dict["proportions"]: proportion of time the individual was in each
            state
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


def statewise_bouts(dataframe):
    """
    Splits a dataframe to generate separate tables for each state
    """

    summary = states_summary(dataframe)
    states = summary["states"]

    statewise_bouts = {}

    for state in states:
        statewise_bouts[state] = dataframe[dataframe["state"] == state].copy()

    return statewise_bouts
 
def fits_to_all_states(dataframe, *args, **kwargs):
    """
    Performs powerlaw.Fit for all states separately
    Args:
        dataframe (pandas.DataFrame): typically yielded by a
        boutparsing.bouts_data_generator()
        args, kwargs: passed on to powerlaw.Fit(...); 
            (see Alstott J, Bullmore E, Plenz D (2014) powerlaw:
            A Python Package for Analysis of Heavy-Tailed Distributions. 
            PLoS ONE 9(1): e85777)
    Returns:
        dict, where for each state,
            dict[state]: powerlaw.Fit object.
    """

    statewise_bouts_data = statewise_bouts(dataframe)
    fitted_distributions = {}

    for state in statewise_bouts_data:
        durations = statewise_bouts_data[state]["duration"]

        if len(durations) < config.minimum_bouts_for_fitting:
            warnings.warn(f"W: insufficient data for state {state}")
            fit = config.insufficient_data_flag
        else:
            fit = pl.Fit(durations, *args, discrete=config.discrete,
                xmin=config.xmin, **kwargs)

        fitted_distributions[state] = fit

    return fitted_distributions


def aic(distribution, data):
    """
    Computes Akaike Information Criteria for a distribution with a dataset
    """

    if type(distribution).__name__ in ['Lognormal', 'Truncated_Power_Law',
                                        'Stretched_Exponential']:
        params = 2
    elif type(distribution).__name__ in ['Exponential', 'Power_Law']:
        params = 1
    else:
        raise ValueError(f"fitting.aic() has not been programmed for\
                            distribution: {distribution}")

    return 2*(params - sum(np.array(distribution.loglikelihoods(data))))

def compare_candidate_distributions(fit, data):
    """
    Computes delta-AICs for all candidate distributions.
    Args:
        fit (powerlaw.Fit): a fit of bout durations.
        data (list like): the data which was used to fit this distribution.
        bootstrap_data (list): bootstraps of given data for comparsison.
        args, kwargs: passed on to powerlaw.Fit()
    Returns:
        pd.DataFrame with col headers=candidate names and values=delta-AICs
    """

    if fit != insufficient_data_flag:
        candidates = config.all_distributions(fit)
        AICs = {}
        dAICs = {}
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
    else:
        dAICs = {}
        for candidate_name in config.distributions_to_numbers:
            dAICs[candidate_name] = [config.insufficient_data_flag]

    return pd.DataFrame(dAICs)

def get_bootstrap_comparisons(all_bootstrap_data, *args, **kwargs):
    data_return = []
    for bootstrap_d, fit in all_bootstrap_data:
        data_return.append(compare_candidate_distributions(fit, bootstrap_d))

    return pd.concat(data_return)

def choose_best_distribution(fit, data):
    """
    Computes the best fitting distribution.
    Args:
        fit (powerlaw.Fit): a fit of bout durations.
        data (list like): the data which was used to fit this distribution.
        args, kwargs: passed on to powerlaw.Fit()
    Returns:
        str: name of the distribution
        fit.<distribution>
    """

    if fit != config.insufficient_data_flag:
        candidates = config.all_distributions(fit)
        min_AIC = np.inf
        best_fit = None
        for candidate_name in candidates:
            candidate = candidates[candidate_name]
            AIC = aic(candidate, data)
            if AIC < min_AIC:
                min_AIC = AIC
                best_fit = candidate_name
        return best_fit, candidates[best_fit]
    #otherwise:
    return config.insufficient_data_flag, config.insufficient_data_flag

def choose_best_bootstrapped_distribution(bootstrap_daics):
    if isinstance(bootstrap_daics, str):
        if bootstrap_daics == config.insufficient_data_flag:
            return config.insufficient_data_flag
    return bootstrap_daics.mean().idxmin()

def print_distribution(dist):
    """
    Returns str-representation of given distribution.
    """

    if dist == config.insufficient_data_flag:
        return config.insufficient_data_flag

    if type(dist) not in [pl.Exponential,
                            pl.Power_Law,
                            pl.Truncated_Power_Law,
                            pl.Lognormal,
                            pl.Stretched_Exponential]:
        raise ValueError(f"Unknown distribution type: {type(dist)}")
        return None

    if type(dist) == pl.Exponential:
        return f"Exponential(λ={dist.Lambda})"
    elif type(dist) == pl.Power_Law:
        return f"Power_Law(α={dist.alpha})"
    elif type(dist) == pl.Truncated_Power_Law:
        return f"Truncated_Power_Law(α={dist.alpha}; λ={dist.Lambda})"
    elif type(dist) == pl.Lognormal:
        return f"Lognormal(μ={dist.mu}; σ={dist.sigma})"
    elif type(dist) == pl.Stretched_Exponential:
        return f"Stretched_Exponential(λ={dist.Lambda}; β={dist.beta})"


def get_ccdf_for_plotting(true_fit, bootstrap_fit):
    def closest_greater_or_equal_index(X, Y):
        # Sort Y and get original indices
        sorted_indices = np.argsort(Y)
        sorted_Y = Y[sorted_indices]
        
        # Find the index of the smallest element in sorted_Y that is >= x
        indices_in_sorted = np.searchsorted(sorted_Y, X, side='left')
        
        # Initialize the result array with -1 (default for no valid index)
        result_indices = np.full_like(X, -1, dtype=int)
        
        # Filter valid indices where indices_in_sorted is within bounds
        valid = indices_in_sorted < len(sorted_Y)
        result_indices[valid] = sorted_indices[indices_in_sorted[valid]]
        
        return result_indices
    true_plot_x, true_plot_y = true_fit.ccdf()
    boots_plot_x, boots_plot_y = bootstrap_fit.ccdf()

    true_plot_x = true_plot_x[:-config.error_bars_rlim]
    boots_plot_x = boots_plot_x[:-config.error_bars_rlim]
    rel_ind = closest_greater_or_equal_index(true_plot_x, boots_plot_x)

    yvals = boots_plot_y[rel_ind]
    yvals[rel_ind == -1] = np.nan
    return true_plot_x, yvals

def make_bootstrap_95_CIs(true_fit, all_bootstrap_data):
    xvals, _ = true_fit.ccdf()
    xvals = xvals[:-config.error_bars_rlim]

    yvals_dat = []
    for boots_data, boots_fit in all_bootstrap_data:
        xvals_temp, yvals_temp = get_ccdf_for_plotting(true_fit, boots_fit)
        assert np.all(np.equal(xvals, xvals_temp))

        yvals_dat.append(yvals_temp)

    yvals_dat = np.array(yvals_dat)
    upper_val = np.nanquantile(yvals_dat, 0.975, method='closest_observation', axis=0)
    lower_val = np.nanquantile(yvals_dat, 0.025, method='closest_observation', axis=0)

    return xvals, upper_val, lower_val


def plot_data_and_fits(fits, state, fig, ax, plot_fits=False, **kwargs):
    """
    Plots cumulative complementary distribution function of data and fitted
    distributions
    Args:
        fits (dict of powerlaw.Fit): typically from fits_to_all_states().
        state (str): behavioural state.
        fig (plt.Figure): figure with ax (below).
        ax (plt.Axes): axis on which to draw.
        plot_fits (bool): whether to plot fitted distributions alongside data.
        **kwargs: passed on to ax.plot(...) via powerlaw.Fit.plot_ccdf(...).
    """

    fit = fits[state]

    if fit != config.insufficient_data_flag:
        fit.plot_ccdf(ax=ax, **kwargs)
        candidate_dists = config.all_distributions(fit)

        if not plot_fits:
            return fig, ax

        for candidate_name in candidate_dists:
            candidate = candidate_dists[candidate_name]
            candidate.plot_ccdf(ax = ax, color=config.colors[candidate_name],
                                    linestyle=config.fit_line_style,
                                    linewidth=0.5,
                                    label=candidate_name)

        return fig, ax
    #otherwise:
    return fig, ax


def test_for_powerlaws(add_bootstrapping=True, add_markov=True):
    """
    Compares candidate distributions and writes to DATA/<species>/<state>.csv
    Also plots distributions.
    Args:
        None
    Returns:
        None
    """

    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    tables = {}
    plots = {}
    bootstrap_info = {}

    print("Initialised distribution fitting sequence.")
    for databundle in bdg:

# Data loading
        print("Processing ", databundle["species"], databundle["id"] + ".")
        data_raw = databundle["data"]
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = boutparsing.as_bouts(data_raw, species_)
 
# Preprocessing
        data = preprocessing_df(data, species_)

# Fitting
        fits = fits_to_all_states(data, verbose=False)
        states = states_summary(data)["states"]

        if species_ not in tables:
            tables[species_] = {}
        if species_ not in plots:
            plots[species_] = {}
        if species_ not in bootstrap_info:
            bootstrap_info[species_] = {}

        for state in states:
            col_names = ["id", "Exponential", "Lognormal",
                            "Power_Law", "Truncated_Power_Law",
                            "Stretched_Exponential", "best_fit"]
            if add_bootstrapping:
                col_names.append("best_fit_bootstrap")

            if state not in tables[species_]:
                tables[species_][state] = pd.DataFrame(columns=col_names)
            if state not in plots[species_]:
                plots[species_][state] = plt.subplots()
            if state not in bootstrap_info[species_]:
                bootstrap_info[species_][state] = []



# Determining best fits
            data_subset = data[data["state"] == state]
            if add_bootstrapping:
                curr_iter_invalid = False
                if len(data_subset["duration"]) < config.minimum_bouts_for_fitting:
                    warnings.warn(f"W: insufficient data in state {state} (add_bootstrapping).")
                    bootstrap_data = config.insufficient_data_flag
                    curr_iter_invalid = True
                else:
                    bootstrap_data = replicates.bootstrapped_data_generator(
                        data_subset["duration"],
                        config.NUM_BOOTSTRAP_REPS
                    )
                    bootstrap_data = list(bootstrap_data)
                    bootstrap_data = [(d,
                        pl.Fit(d, discrete=config.discrete,
                                xmin=config.xmin)
                        ) for d in bootstrap_data]

            table = compare_candidate_distributions(fits[state],
                                                    data_subset["duration"])
            table["id"] = databundle["id"]
            _, best_dist = choose_best_distribution(fits[state],
                                    data_subset["duration"])
            table["best_fit"] = print_distribution(best_dist)

            if add_bootstrapping:
                if curr_iter_invalid:
                    big_bunch_of_daics = bootstrap_data
                else:
                    big_bunch_of_daics = get_bootstrap_comparisons(bootstrap_data)
                table["best_fit_bootstrap"] =\
                    choose_best_bootstrapped_distribution(big_bunch_of_daics)
            if not tables[species_][state].empty:
                tables[species_][state] = pd.concat([tables[species_][state], table])
            else:
                tables[species_][state] = table

# Generating figures
            fig, ax = plots[species_][state]
            plots[species_][state] = plot_data_and_fits(fits, state, fig, ax,
                                                        plot_fits=False,
                                                        color="darkred",
                                                        alpha=0.3)
            if add_bootstrapping:
                if not curr_iter_invalid:
                    xs, upper_lim, lower_lim = make_bootstrap_95_CIs(fits[state], bootstrap_data)
                    ax.fill_between(xs, upper_lim, lower_lim,
                                    color="darkred", alpha=0.09)



# Saving tabulate data
    print("Generating tables and plots.")
    os.makedirs(os.path.join(config.DATA, "FitResults"), exist_ok=True)
    for species in tables:
        for state in tables[species]:
            tables[species][state].to_csv(os.path.join(config.DATA,
                                                    "FitResults", species,
                                                    state + ".csv"),
                                                index=False)

# Saving figures
    for species in plots:
        for state in plots[species]:
            fig, ax = plots[species][state]
            epoch = classifier_info.classifiers_info[species].epoch
            if classifier_info.classifiers_info[species].epoch != 1.0:
                ax.set_xlabel(f"Time ($\\times {epoch}$ seconds)")
            else:
                ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("CCDF")
#            ax.set_title(f"Species: {species.title()} | State: {state.title()}")
            utilities.saveimg(fig, f"Distribution-fits-{species}-{state}")

# Done
    print("Distribution fitting completed.")

if __name__ == "__main__":
    if config.COLLAGE_IMAGES:
        plt.rcParams.update({'font.size': 22})

    test_for_powerlaws()
