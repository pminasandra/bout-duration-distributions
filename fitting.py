# Pranav Minasandra
# pminasandra.github.io
# Dec 26, 2022

import os.path

import numpy as np
import powerlaw as pl

import config
import boutparsing
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

def states_summary(dataframe):
    """
    Finds all the states and provides a basic quantitative summary of these states.
    Args:
        dataframe (pd.DataFrame): typically yielded by a boutparsing.bouts_data_generator()
    Returns:
        dict, where
            dict["states"]: list of all states whose bouts are in states summary
            dict["proportions"]: proportion of time the individual was in each state
    """

