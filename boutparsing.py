# Pranav Minasandra
# pminasandra.github.io
# Dec 25, 2022

"""
This module provides the following generators to use for data retrieval from all three species.
<under construction>
"""

import glob
import os.path
import warnings

import pandas as pd

import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

hyena_dir = os.path.join(config.DATA, "hyena")
meerkat_dir = os.path.join(config.DATA, "meerkat")
#TODO: coati_dir = os.path.join(config.DATA, coati)


def hyena_data_generator():
    """
    *GENERATOR* yields behavioural sequence data for hyenas, individual-by-individual.
    Args:
        None so far.
    Yields:
        dict, where
            dict["data"]: pd.DataFrame
            dict["id"]: str, identifying information for the individual
    """

    for hyena in glob.glob(os.path.join(hyena_dir, "*.csv")):
        name = os.path.basename(hyena)[:-len(".csv")]
        read = pd.read_csv(hyena, header=0)
        yield {"data": read, "id": name}
