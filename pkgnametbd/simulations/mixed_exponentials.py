# Pranav Minasandra
# pminasandra.github.io
# Sep 00, 1013

import random as random

import numpy as np
import powerlaw as pl

from pkgnametbd import config
from pkgnametbd import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

class MixedExponential:
    """
    Class that mimics distribtions in the module powerlaw
    Acts as a mixture of two exponential distributions
    """
    def __init__(self, ps, lambdas, xmin=config.xmin, discrete=config.discrete):
        """
        Args:
            ps (array-like): weights for distributions
            lambdas (array-like): parameters for each distribution
        """

        assert np.isclose(sum(ps), 1.0)

        self.weights = np.array(ps)
        self.dists = np.array([pl.Exponential(xmin=xmin, parameters=[l],
                        discrete=discrete) for l in lambdas])

    def generate_random(self, size):

        num_dists = len(self.weights)
        indices = np.arange(num_dists)
        num_draws = np.random.choice(indices, size=size, 
                                    p=self.weights, replace=True)

        random_vars = []
        for index in indices:
            count = len(num_draws[num_draws==index])
            a = self.dists[index].generate_random(n=count)
            random_vars.extend(list(a))

        return random_vars
