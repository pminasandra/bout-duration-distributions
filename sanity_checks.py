# Pranav Minasandra
# pminasandra.github.io
# Dec 31, 2022
# Nice way to end the year, with some sanity checks

import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import boutparsing
import fitting
import config
import utilities

old_print = print
if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def check_state_consistency():
    bdg = boutparsing.bouts_data_generator()
    states = {}
    for databundle in bdg:
        species = databundle["species"]
        print("loading data for", species, databundle["id"])
        if species not in states:
            states[species] = []
        states_, props_ = fitting.states_summary(databundle["data"])
        states[species].append(states_)

    for species in states:
        print("states check for", species)
        for i in range(len(states[species]) - 1):
            assert states[species][i] == states[species][i+1]

    print("check_state_consistency: passed.")

def check_contextness():
    """
    Provides a way to look for how context-driven behavioural bouts are, by
    checking how much consecutive bouts for the same behaviour are correlated.
    """

    import scipy.stats
    from matplotlib.offsetbox import AnchoredText

    print("contextness check initiated.")
    bdg = boutparsing.bouts_data_generator()
    states = {}
    statistics = {}

    for databundle in bdg:
        species = databundle["species"]
        id_ = databundle["id"]
        data = fitting.preprocessing_df(databundle["data"], species)
        print("loading data for", species, databundle["id"])
        if species not in states:
            states[species] = {}
            statistics[species] = {}
        statewise_bouts = fitting.statewise_bouts(data)

        for state in statewise_bouts:
            if state not in states[species]:
                states[species][state] = plt.subplots()
                statistics[species][state] = [[], [], []] #id, r, p

            points = np.array(statewise_bouts[state]["duration"])

            fig, ax = states[species][state]
            ax.scatter(points[:-1], points[1:], s=0.1)
            ax.set_title(f"Species: {species.title()} | State: {state.title()}")

            correlation_test_results = scipy.stats.spearmanr(points[:-1], points[1:])
            statistics[species][state][0].append(id_)
            statistics[species][state][1].append(correlation_test_results.statistic)
            statistics[species][state][2].append(correlation_test_results.pvalue)

    print("generating images and saving data.")
    for species in states:
        os.makedirs(os.path.join(config.DATA, "SanityCheckResults/", species), exist_ok=True)
        for state in states[species]:
            fd = open(os.path.join(config.DATA, "SanityCheckResults/", species, "contextness_" + state + ".csv"), "w")
            old_print("id,r,p", file=fd)
            fd.close()

        for state in states[species]:
            fd = open(os.path.join(config.DATA, "SanityCheckResults/", species, "contextness_" + state + ".csv"), "a")
            fig, ax = states[species][state]
            utilities.saveimg(fig, f"subsequent_bouts_correlation_{species}_{state}")

            stats = statistics[species][state]
            for id_, r, p in zip(*stats): 
                old_print(id_, r, p, sep=",", file=fd) #print straight to file, suppressing file_info from utilities.sprint
        fd.close()
        #states[species].append(states_)
    print("contextness check completed.")


def generate_data_summary(datagen):
    """
    Generates a summary of available data and the bouts therein
    Args:
        datagen: a generator, typically from boutparsing.bouts_data_generator()
    """

    data_summaries = {}
    for databundle in datagen:
        species = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]

        print(f"generate_data_summary: parsing {species} {id_}.")
        if species not in data_summaries:
            data_summaries[species] = {}
        data_summaries[species][id_] = {}
        data_summaries[species][id_]['data_available'] = len(data['state'])/3600.0
        data_summaries[species][id_]['days_available'] =\
                                len(data['datetime'].dt.date.unique())

        data = boutparsing.as_bouts(data, species)
        statewise_bouts = fitting.statewise_bouts(data)
        for state in statewise_bouts:
            data_summaries[species][id_][state.title()] =\
                                len(statewise_bouts[state])

    for species in data_summaries:
        data_summ_df = pd.DataFrame(data_summaries[species])
        data_summ_df = data_summ_df.T
        for col in data_summ_df.columns:
            if col != "data_available":#Stupid bug in new version of pandas
                data_summ_df[col] = data_summ_df[col].astype(int)
        data_summ_df['days_available'] =\
                data_summ_df['days_available'].astype(int)
        print(species, '\n', data_summ_df)
        fd = open(os.path.join(config.DATA, f"data-summary-{species}.tex"),
                                                                        "w")
        old_print(data_summ_df.to_latex(index=False,
                                    float_format="{:.3f}".format),
                                file = fd)
        fd.close()



if __name__ == "__main__":
    #check_state_consistency()
    #check_contextness()

    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    generate_data_summary(bdg)
