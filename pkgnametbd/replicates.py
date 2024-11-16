# Pranav Minasandra
# pminasandra.github.io
# 16 Oct 2024

"""
Provides functions needed to bootstrap data and to
make Markov-assumption replicates of behavioural sequences.
"""

import os
import os.path

import numpy as np
import pandas as pd

import boutparsing
import config
import classifier_info
import fitting
import persistence
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint


def bootstrap(bouts, size=None):
    """
    Draws 'size (int)' bouts from the given dataset 'bouts'
    with replacement.
    """

    if size is None:
        size = len(bouts)

    return np.random.choice(bouts, size=size)


def bootstrap_iter(bouts, num_replicates, size=None):
    """
    bootstrap(...) object wrapped in a convenient iterator
    Args:
        bouts (list-like): bout length data
        num_replicates (int): number of replicates
        size (int): num samples to draw per replicate
    """

    for i in range(num_replicates):
        yield bootstrap(bouts, size=size)


def markovian_sequence(beh_seq, species, length=None, start=None):
    """
    Taking in a behavioural sequence, returns a different sequence
    that retains the first-order (e.g., second-to-second) memory, but
    loses memory at longer time-scales.
    Args:
        beh_seq (pd.DataFrame): complete behavioural sequence.
        s[ecies (str): name of species.
        length (int): how long the generated markovian sequence must be.
                None implies same as len(beh_seq).
        start (any): the state to start from. default None implies the same
                starting as beh_seq. typeof(start) must be same as that of
                elements in beh_seq["state"].
    """

    if length is None:
        length = len(beh_seq["state"])
    if start is None:
        start = beh_seq["state"][0]

    # create the necessary piles for drawing with replacement
    piles = {}
    epoch = classifier_info.classifiers_info[species].epoch
    states = beh_seq["state"].unique()
    seq_len = len(beh_seq["state"])

    for state in states:
        ind_state = beh_seq[beh_seq["state"] == state].index
        ind_state = ind_state[ind_state < seq_len - 1]
        ind_relevant = ind_state + 1
        valid = persistence._validated_paired_indices(beh_seq["datetime"],
                            1,
                            ind_state,
                            ind_relevant,
                            epoch)
        ind_relevant = ind_relevant[valid]
        ind_relevant = ind_relevant[ind_relevant < seq_len]

        piles[state] = beh_seq["state"][ind_relevant].to_numpy()

    seq = []
    curr_state = start
    for t in range(length):
        seq.append(curr_state)
        curr_state = np.random.choice(piles[curr_state])

    dts = np.arange(0, length, 1)
    dts = pd.to_datetime(dts*1e9) #dummy datetimes for other functions
    df = pd.DataFrame({"datetime": dts, "state": seq})

    return df

if __name__ == "__main__":
    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]

        k = markovian_sequence(data, species_)
        print(boutparsing.as_bouts(k, "meerkat"))
        break
