# Pranav Minasandra
# pminasandra.github.io
# 16 Oct 2024

"""
Provides functions needed to bootstrap data and to
make Markov-assumption replicates of behavioural sequences.
"""

import glob
import multiprocessing as mp
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


def bootstrapped_data_generator(bouts, num_replicates, size=None):
    """
    *GENERATOR*
    bootstrap(...) object wrapped in a convenient generator
    Args:
        bouts (list-like): bout length data
        num_replicates (int): number of replicates
        size (int): num samples to draw per replicate
    """

    for i in range(num_replicates):
        yield bootstrap(bouts, size=size)


def markovised_sequence(beh_seq, species, length=None, start=None):
    """
    Taking in a behavioural sequence, returns a different sequence
    that retains the first-order (e.g., second-to-second) memory, but
    loses memory at longer time-scales.
    Args:
        beh_seq (pd.DataFrame): complete behavioural sequence.
        species (str): name of species.
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

    dts = np.arange(0, length, 1).astype(float)
    dts *= epoch
    dts = pd.to_datetime(dts*1e9) #dummy datetimes for other functions
    df = pd.DataFrame({"datetime": dts, "state": seq})

    return df

def markovised_sequence_generator(beh_seq, species, num_replicates, length=None, start=None):
    """
    *GENERATOR*
    markovised_sequence(...) wrapped in a convenient generator
    Args:
        beh_seq (pd.DataFrame): complete behavioural sequence.
        species (str): name of species.
        num_replicates (int): number of replicates
        length (int): how long the generated markovian sequence must be.
                None implies same as len(beh_seq).
        start (any): the state to start from. default None implies the same
                starting as beh_seq. typeof(start) must be same as that of
                elements in beh_seq["state"].
    """
    for i in range(num_replicates):
        yield markovised_sequence(beh_seq, species, length=length, start=start)

def load_markovisations(species_, id_):
    """
    Load all markovisations for the organism specified by
    species_ and id_ (both str)
    """
    i = 0
    tgtdir = os.path.join(config.DATA, "Markovisations", species_, id_)

    files = glob.glob(os.path.join(tgtdir, "*.csv"))
    for filename in files:
        if i >= config.NUM_MARKOVISED_SEQUENCES:
            break
        df = pd.read_csv(filename)
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
        i += 1
        yield df

def _dummyfunc(x):
    return x

def load_markovisations_parallel(species_, id_):
    """
    Parallely load up all markoivsations, because file-reading
    at these sizes takes forever. 
    Args:
        species_ and id_ (both str)
    """
    pool = mp.Pool()
    gen = load_markovisations(species_, id_)
    mdatasets = pool.map(_dummyfunc, gen)
    pool.close()
    pool.join()

    return mdatasets

if __name__ == "__main__":
    print("Generating Markovised Sequences.")

# Make sure all necessary directories are set up
    markovdir = os.path.join(config.DATA, "Markovisations")
    os.makedirs(markovdir, exist_ok=True)

    for species_ in config.species:
        os.makedirs(os.path.join(markovdir, species_), exist_ok=True)

# Create and save Markovisations
    bdg = boutparsing.bouts_data_generator(extract_bouts=False)
    for databundle in bdg:
        species_ = databundle["species"]
        id_ = databundle["id"]
        data = databundle["data"]

        tgtdir = os.path.join(markovdir, species_, id_)
        os.makedirs(tgtdir, exist_ok=True)
        datagen = markovised_sequence_generator(data,
                                species_, config.NUM_MARKOVISED_SEQUENCES)
        for i, dataset in enumerate(datagen):
            print(f"Generating Markovised sequences for",
                    f"{species_} {id_}: {i+1} out of",
                    f"{config.NUM_MARKOVISED_SEQUENCES}")
            dataset.to_csv(os.path.join(tgtdir, f"{i}.csv"),
                                    index=False)
