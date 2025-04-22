# Pranav Minasandra
# pminasandra.github.io
# Dec 25, 2022

"""
This module provides the following generators to use for data retrieval from all
three species.
"""

import datetime as dt
import glob
import os.path
import random

import pandas as pd

from pkgnametbd import classifier_info
from pkgnametbd import config
from pkgnametbd import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint


def as_bouts(dataframe, species, randomize=False):
    """
    Reduces time-stamped behaviour sequences into bout information.
    """

    epoch = classifier_info.classifiers_info[species].epoch
    datetimes = dataframe["datetime"].tolist()
    states = dataframe["state"].tolist()

    if randomize:
        random.shuffle(states)

    current_state = "UNKNOWN"
    previous_state = "UNKNOWN"
    previous_recorded_state = "UNKNOWN"
    state_duration = 0.0
    row_num = 0

    bout_states = []
    bout_durations = []

    for datetime, state in zip(datetimes, states):
        if current_state == "UNKNOWN":
        # Can only happen in first iteration
            row_num += 1
            previous_state = states[0]
            current_state = state
            continue
        current_state = state
        if datetime - datetimes[row_num - 1] != dt.timedelta(seconds=epoch):
        # If there's a gap in the sequence
            previous_recorded_state = "UNKNOWN"
            previous_state = "UNKNOWN"
            state_duration = 0.0
        else:
            state_duration += epoch
        current_state = state

        if current_state != previous_state:
        # State changed, so update records
            if previous_recorded_state != "UNKNOWN" and previous_state != "UNKNOWN":
                # If the state just now was unknown (timeskip) or
                # if the previous state was not known
                bout_states.append(previous_state)
                bout_durations.append(state_duration)
            previous_recorded_state = previous_state
            state_duration = 0.0

        previous_state = state
        row_num += 1

    boutdf = pd.DataFrame([bout_durations, bout_states]).T
    boutdf.columns = ["duration", "state"]
    boutdf = boutdf.infer_objects()
    return boutdf

def default_datagen_creator(species):
    """
    Creates generator functions to iteratively extract data from each species
    Args:
        species (str): species name, AND ALSO data directory name. ENSURE that
                        these are the same.
    """

    def data_generator(randomize=False, extract_bouts=True):
        f"""
        *GENERATOR* yields behavioural sequence data and metadata from {species},
        individual-by-individual.
        Args:
            randomize (bool): whether to randomize data before extracting bouts.
        Yields:
            dict, where
                dict["data"]: pd.DataFrame
                dict["id"]: str, identifying information for the individual
                dict["species"]: str, species of the individual whose data is in
                                dict["data"]
        """
        if not extract_bouts:
            def postproc(*args, **kwargs):
                return args[0]
        else:
            postproc = as_bouts

        for ind in glob.glob(os.path.join(config.DATA, species, "*.csv")):
            name = os.path.basename(ind)[:-len(".csv")]
            read = pd.read_csv(ind, header=0)
            read["datetime"] = pd.to_datetime(read["datetime"], format="mixed")
            yield {
                   "data": postproc(read, species, randomize=randomize),
                   "id": name,
                   "species": species
                  }

    return data_generator

meerkat_data_generator = default_datagen_creator("meerkat")
coati_data_generator = default_datagen_creator("coati")
hyena_data_generator = default_datagen_creator("hyena")
hyrax_data_generator = default_datagen_creator("hyrax")

generators = {
                "meerkat": meerkat_data_generator,
                "coati": coati_data_generator,
                "hyena": hyena_data_generator,
                "hyrax": hyrax_data_generator
            }

def bouts_data_generator(randomize=False, extract_bouts=True):
    """
    *GENERATOR* yields behavioural sequence data and metadata for all species,
    individual-by-individual,
    Args:
        randomize (bool): whether to randomize data before extracting bouts.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
            dict["species"]: str, species of the individual whose 
                            data is in dict["data"]
    """
    for species in config.species:
        datasource = generators[species](randomize=randomize, extract_bouts=extract_bouts)
        for databundle in datasource:
                yield databundle

