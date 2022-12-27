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
#TODO: coati_dir = os.path.join(config.DATA, coati)


def as_bouts(dataframe, species):
    """
    Reduces time-stamped behaviour sequences into bout information.
    """

    epoch = classifier_info.classifiers_info[species].epoch
    datetimes = dataframe["datetime"].tolist()
    states = dataframe["state"].tolist()

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
            bout_states.append(previous_state)
            bout_durations.append(state_duration)
            state_duration = 0.0

        previous_state = state
        row_num += 1

    boutdf = pd.DataFrame([bout_durations, bout_states]).T
    boutdf.columns = ["duration", "state"]
    return boutdf


def hyena_data_generator():
    """
    *GENERATOR* yields behavioural sequence data and metadata for hyenas, individual-by-individual.
    Args:
        None so far.
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
               "data": as_bouts(read, "hyena"),
               "id": name,
               "species": "hyena"
              }

#TODO:
def meerkat_data_generator():
    return
#TODO:
#def coati_data_generator()

generators = {"hyena": hyena_data_generator, "meerkat": meerkat_data_generator}
def bouts_data_generator():
    """
    *GENERATOR* yields behavioural sequence data and metadata for all species, individual-by-individual,
    Args:
        None so far
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose data is in dict["data"]
    """
    for species in ["hyena"]:#FIXME config.species:
        datasource = generators[species]()
        for databundle in datasource:
                yield databundle

hdg = hyena_data_generator()
for db in hdg:
    print(db["id"], db["data"], "\n\n\n")
