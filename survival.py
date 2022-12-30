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
    epoch = classifier_info.classifiers_info[species].epoch

    dataframe = dataframe.copy()
    dataframe["duration"] /= epoch

    dataframe = dataframe[dataframe["duration"] >= config.xmin]
    dataframe = dataframe[dataframe["state"] == state]

    unique_values = dataframe["duration"].unique()
    unique_values.sort()
    unique_values = unique_values[:-1] #Excluding the biggest bout because it contributes nothing here

    ts = []
    BIs = []

    for t in unique_values:
        BI_main = dataframe[dataframe["duration"] > t]
        BI_denom = BI_main["duration"].count()
        BI_sub = dataframe[dataframe["duration"] > (t + 1.0)] 
        BI_numer = BI_sub["duration"].count()

        ts.append(t)
        BIs.append(BI_numer/BI_denom)

    table = np.array([ts, BIs]).T
    return table

def generate_behavioural_inertia_plots():

    print("Behavioural inertia plot generation initiated.")
    bdg = boutparsing.bouts_data_generator()

    plots = {}
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]
        print(f"Working on {id_}.")

        if species_ not in plots:
            plots[species_] = {}

        states = fitting.states_summary(data)["states"]

        for state in states:
            if state not in plots[species_]:
                plots[species_][state] = plt.subplots()

            survival_table = compute_behavioural_inertia(data, species_, state)
            fig, ax = plots[species_][state]
            ax.step(survival_table[:,0], survival_table[:,1], color=config.survival_plot_color, linewidth=0.75, alpha=0.4)

    print("Data analysis completed, saving plots.")
    for species in plots:
        epoch = classifier_info.classifiers_info[species].epoch

        for state in plots[species]:
            fig, ax = plots[species][state]
            ax.set_xscale("log")

            if epoch != 1.0:
                ax.set_xlabel(f"Time ($\\times {epoch}$ seconds)")
            else:
                ax.set_xlabel("Time (seconds)")

            ax.set_ylabel("Behavioural Inertia")
            ax.set_title(f"Species: {species.title()} | State: {state.title()}")

            utilities.saveimg(fig, f"Behavioural-Inertia-{species}-{state}")

    print("Behavioural inertia plot generation finished.")

if __name__ == "__main__":
    generate_behavioural_inertia_plots()
