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

import config
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
