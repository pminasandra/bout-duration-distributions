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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw as pl

import config
import classifier_info
import boutparsing
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def preprocessing_df(dataframe, species):
    """
    Converts bouts to epoch units.
    Ensures that the bouts of 1 epoch are removed, and any other preprocessing decided later.
    """

    df = dataframe.copy()
    df["duration"] /= classifier_info.classifiers_info[species].epoch
    df = df[df["duration"] >= config.xmin]

    return df


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
        dataframe (pandas.DataFrame): typically yielded by a boutparsing.bouts_data_generator()
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

        fit = pl.Fit(durations, *args, discrete=config.discrete, xmin=config.xmin, **kwargs)

        fitted_distributions[state] = fit

    return fitted_distributions


def aic(distribution, data):
    if type(distribution).__name__ in ['Lognormal', 'Truncated_Power_Law', 'Stretched_Exponential']:
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

    return pd.DataFrame(dAICs)

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

def print_distribution(dist):
    """
    Returns str-representation of given distribution.
    """

    if type(dist) not in [pl.Exponential, pl.Power_Law, pl.Truncated_Power_Law, pl.Lognormal, pl.Stretched_Exponential]:
        raise ValueError(f"Unknown distribution type: {type(dist)}")

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

def plot_data_and_fits(fits, state, fig, ax, plot_fits=False, **kwargs):
    """
    Plots cumulative complementary distribution function of data and fitted distributions
    Args:
        fits (dict of powerlaw.Fit): typically from fits_to_all_states().
        state (str): behavioural state.
        fig (plt.Figure): figure with ax (below).
        ax (plt.Axes): axis on which to draw.
        plot_fits (bool): whether to plot fitted distributions alongside data.
        **kwargs: passed on to ax.plot(...) via powerlaw.Fit.plot_ccdf(...).
    """

    fit = fits[state]
    fit.plot_ccdf(ax=ax, **kwargs)
    candidate_dists = config.all_distributions(fit)

    if not plot_fits:
        return fig, ax

    for candidate_name in candidate_dists:
        candidate = candidate_dists[candidate_name]
        candidate.plot_ccdf(ax = ax, color=config.colors[candidate_name], linestyle=config.fit_line_style, linewidth=0.5, label=candidate_name)

    return fig, ax

def test_for_powerlaws():
    """
    Compares candidate distributions and writes to DATA/<species>/<state>.csv
    Also plots distributions.
    Args:
        None
    Returns:
        None
    """

    bdg = boutparsing.bouts_data_generator()
    tables = {}
    plots = {}

    print("Initialised distribution fitting sequence.")
    for databundle in bdg:
        print("Processing ", databundle["species"], databundle["id"] + ".")
        data = databundle["data"]
        species_ = databundle["species"]
        
        data = preprocessing_df(data, species_)

        fits = fits_to_all_states(data, verbose=False)
        states = states_summary(data)["states"]

        if species_ not in tables:
            tables[species_] = {}
        if species_ not in plots:
            plots[species_] = {}

        for state in states:
            if state not in tables[species_]:
                tables[species_][state] = pd.DataFrame(columns=["id", "Exponential", "Lognormal", "Power_Law", "Truncated_Power_Law", "Stretched_Exponential", "best_fit"])
            if state not in plots[species_]:
                plots[species_][state] = plt.subplots()

            table = compare_candidate_distributions(fits[state], data["duration"])
            table["id"] = databundle["id"]
            _, best_dist = choose_best_distribution(fits[state], data["duration"])
            table["best_fit"] = print_distribution(best_dist)
            tables[species_][state] = pd.concat([tables[species_][state], table])

            fig, ax = plots[species_][state]
            plots[species_][state] = plot_data_and_fits(fits, state, fig, ax, plot_fits=False, color="darkred", alpha=0.3)


    print("Generating tables and plots.")
    for species in tables:
        for state in tables[species]:
            tables[species][state].to_csv(os.path.join(config.DATA, "FitResults", species, state + ".csv"), index=False)

    for species in plots:
        for state in plots[species]:
            fig, ax = plots[species][state]
            epoch = classifier_info.classifiers_info[species].epoch
            if classifier_info.classifiers_info[species].epoch != 1.0:
                ax.set_xlabel(f"Time ($\\times {epoch}$ seconds)")
            else:
                ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("$\\Pr\\left(\\tau \\ge t \\right)$")
            ax.set_title(f"{species.title()} {utilities.lowerfirst(config.PRETTY_STATE_LABELS[state])}")
            utilities.saveimg(fig, f"Distribution-fits-{species}-{state}")

    print("Distribution fitting completed.")

if __name__ == "__main__":
    test_for_powerlaws()
