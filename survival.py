# Pranav Minasandra
# pminasandra.github.io
# Dec 29, 2022

"""
This module provides the function compute_behavioural_inertia, which computes the complement of the hazard function
based on a given dataset and state. It also includes the wrapper function generate_behavioural_inertia_plots(),
which does this analysis on all species, individuals, and behavioural states. 
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

import boutparsing
import config
import classifier_info
import fitting
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

# NOTE: The below function has an argument, hazard_rate, that defaults to False.
# Initially, we thought 1 - Hazard was a better pitch for the paper. However,
# since, the paper has switched to hazard rate in totality. Make sure to always
# turn hazard_rate to False whenever using this function.
def compute_behavioural_inertia(dataframe, species, state, hazard_rate=False):
    """
    Computes behavioural inertia (see docs) for a given dataset.
    Args:
        dataframe (pd.DataFrame): bout information, typically output from boutparsing.bouts_data_generator(...).
        species (str): name of species from which the data come.
        state (str): name of the behavioural state.
        hazard_rate (bool): default False, whether to plot hazard rate instead of behavioural inertia
    Returns:
        np.array (two column): time and inertia values
   """
    xmin = config.xmin
    epoch = classifier_info.classifiers_info[species].epoch

    dataframe = fitting.preprocessing_df(dataframe, species)
    dataframe = dataframe[dataframe["state"] == state]

    unique_values = dataframe["duration"].unique()
    unique_values.sort()
    unique_values = unique_values[:-1] #Excluding the biggest bout because it contributes nothing here
    if config.survival_exclude_last_few_points:
        if len(dataframe["duration"]) < config.survival_num_points_to_exclude:
            return "invalid" # sort of hacky, unfortunately :(
        sorted_vals = np.sort(dataframe["duration"])
        nth_from_last = sorted_vals[-config.survival_num_points_to_exclude]

        unique_values = unique_values[unique_values < nth_from_last]
    ts = []
    BIs = []

    for t in unique_values:
        BI_main = dataframe[dataframe["duration"] > t]
        BI_denom = BI_main["duration"].count()
        BI_sub = dataframe[dataframe["duration"] > (t + 1.0)] 
        BI_numer = BI_sub["duration"].count()

        ts.append(t)
        if not hazard_rate:
            BIs.append(BI_numer/BI_denom)
        else:
            BIs.append(1 - (BI_numer/BI_denom))

    table = np.array([ts, BIs]).T
    return table

def generate_behavioural_inertia_plots(add_randomized=False, hazard_rate=False):
    """
    Generates behavioural inertia vs time plots for all states, individuals, species.
    Args:
        add_randomized (bool): default False, whether to also add same plots with shuffling of data
        hazard_rate (bool): default False, whether to plot hazard rate instead of behavioural inertia
    """
    print("Behavioural inertia plot generation initiated.")
    bdg = boutparsing.bouts_data_generator()
    if add_randomized:
        bdg_r = boutparsing.bouts_data_generator(randomize=True)

    plots = {}
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]
        print(f"Working on {species_} {id_}.")

        if species_ not in plots:
            plots[species_] = {}

        states = fitting.states_summary(data)["states"]

        for state in states:
            if state not in plots[species_]:
                plots[species_][state] = plt.subplots()

            survival_table = compute_behavioural_inertia(data, species_, state, hazard_rate=hazard_rate)
            if isinstance(survival_table, str):
                if survival_table == "invalid":
                    continue
            fig, ax = plots[species_][state]
            ax.step(survival_table[:,0], survival_table[:,1], color=config.survival_plot_color, linewidth=0.75, alpha=0.4)
            ax.set_xscale(config.survival_xscale)
            ax.set_yscale(config.survival_yscale)

    if add_randomized:
        for databundle in bdg_r:
            species_ = databundle["species"]
            id_ = databundle["id"]
            data = databundle["data"]
            print(f"Working on randomised data for {species_} {id_}.")

            if species_ not in plots:
                plots[species_] = {}

            states = fitting.states_summary(data)["states"]

            for state in states:
                if state not in plots[species_]:
                    plots[species_][state] = plt.subplots()

                survival_table = compute_behavioural_inertia(data, species_, state, hazard_rate=hazard_rate)
                fig, ax = plots[species_][state]
                ax.step(survival_table[:,0], survival_table[:,1], color=config.survival_randomization_plot_color, linewidth=0.75, alpha=0.4)

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

            ax.set_title(f"Species: {species.title()} | State: {state.title()}")
            if hazard_rate:
                ax.set_ylabel("Hazard function")
                utilities.saveimg(fig, f"Hazard-Rate-{species}-{state}")
            else:
                ax.set_ylabel("Behavioural Inertia")
                utilities.saveimg(fig, f"Behavioural-Inertia-{species}-{state}")



    print("Behavioural inertia plot generation finished.")

if __name__ == "__main__":
    generate_behavioural_inertia_plots(add_randomized=False, hazard_rate=True)
