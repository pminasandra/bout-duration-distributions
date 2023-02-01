# Pranav Minasandra
# pminasandra.github.io
# Dec 25, 2022

"""
This module provides the following generators to use for data retrieval from all three species.
<under construction>
"""

import datetime as dt
import glob
import os.path
import warnings

import pandas as pd

import classifier_info
import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

hyena_dir = os.path.join(config.DATA, "hyena")
meerkat_dir = os.path.join(config.DATA, "meerkat")
coati_dir = os.path.join(config.DATA, "coati")


def as_bouts(dataframe, species, randomize=False):
    """
    Reduces time-stamped behaviour sequences into bout information.
    """

    epoch = classifier_info.classifiers_info[species].epoch
    datetimes = dataframe["datetime"].tolist()
    states = dataframe["state"].tolist()

    if randomize:
        import random
        random.shuffle(states)

    current_state = "UNKNOWN"
    previous_state = "UNKNOWN"
    state_duration = 0.0
    row_num = 0

    bout_states = []
    bout_durations = []

    for datetime, state in zip(datetimes, states):
        if current_state == "UNKNOWN":
            row_num += 1
            previous_state = states[0]
            current_state = state
            continue
        current_state = state
        if datetime - datetimes[row_num - 1] != dt.timedelta(seconds=epoch):
            previous_state = "UNKNOWN"
            state_duration = 0.0
        else:
            state_duration += epoch
        current_state = state

        if current_state != previous_state:
            if previous_state != "UNKNOWN":
                bout_states.append(previous_state)
                bout_durations.append(state_duration)
            state_duration = 0.0

        previous_state = state
        row_num += 1

    boutdf = pd.DataFrame([bout_durations, bout_states]).T
    boutdf.columns = ["duration", "state"]
    boutdf = boutdf.infer_objects()
    return boutdf


def hyena_data_generator(randomize=False):
    """
    *GENERATOR* yields behavioural sequence data and metadata for hyenas, individual-by-individual.
    Args:
        randomize (bool): whether to randomize data before extracting bouts.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose data is in dict["data"]
    """

    for hyena in glob.glob(os.path.join(hyena_dir, "*.csv")):
        name = os.path.basename(hyena)[:-len(".csv")]
        read = pd.read_csv(hyena, header=0)
        read["datetime"] = pd.to_datetime(read["datetime"])
        yield {
               "data": as_bouts(read, "hyena", randomize=randomize),
               "id": name,
               "species": "hyena"
              }

def meerkat_data_generator(randomize=False):
    """
    *GENERATOR* yields behavioural sequence data and metadata for meerkats, individual-by-individual.
    Args:
        randomize (bool): whether to randomize data before extracting bouts.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose data is in dict["data"]
    """

    for meerkat in glob.glob(os.path.join(meerkat_dir, "*/*.csv")):
        name = os.path.basename(meerkat)[:-len(".csv")]
        read = pd.read_csv(meerkat, header=0)
        read["datetime"] = pd.to_datetime(read["datetime"])
        yield {
               "data": as_bouts(read, "meerkat", randomize=randomize),
               "id": name,
               "species": "meerkat"
              }
def coati_data_generator(randomize=False):
    """
    *GENERATOR* yields behavioural sequence data and metadata for coatis, individual-by-individual.
    Args:
        randomize (bool): whether to randomize data before extracting bouts.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose data is in dict["data"]
    """

    for coati in glob.glob(os.path.join(coati_dir, "*.csv")):
        name = os.path.basename(coati)[:-len(".csv")]
        read = pd.read_csv(coati, header=0)
        read["datetime"] = pd.to_datetime(read["datetime"])
        yield {
               "data": as_bouts(read, "coati", randomize=randomize),
               "id": name,
               "species": "coati"
              }

generators = {
                "hyena": hyena_data_generator,
                "meerkat": meerkat_data_generator,
                "coati": coati_data_generator
            }

def bouts_data_generator(randomize=False):
    """
    *GENERATOR* yields behavioural sequence data and metadata for all species, individual-by-individual,
    Args:
        randomize (bool): whether to randomize data before extracting bouts.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose data is in dict["data"]
    """
    for species in config.species:
        datasource = generators[species](randomize=randomize)
        for databundle in datasource:
                yield databundle

