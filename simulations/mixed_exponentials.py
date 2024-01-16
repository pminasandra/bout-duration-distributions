# Pranav Minasandra
# pminasandra.github.io
# Sep 00, 1013

import random as random

import numpy as np
import powerlaw as pl

import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

class MixedExponential:
    """
    Class that mimics distribtions in the module powerlaw
    Acts as a mixture of two exponential distributions
    """
    def __init__(self, p_0, exp_lambda_0, exp_lambda_1, xmin=config.xmin, discrete=config.discrete):
        """
        Args:
            p_0 (float): weight for first distribution (Note that p_1 is inherently 1 - p_0)
            exp_lambda_0 and exp_lambda_1 (float, >0): parameters for the two exponential distributions
        """

        assert p_0 >= 0
        assert 1 - p_0 >= 0
        assert exp_lambda_0 > 0
        assert exp_lambda_1 > 0

        self.p_0 = p_0
        self.p_1 = 1 - p_0
        self.exp_lambda_0 = exp_lambda_0
        self.exp_lambda_1 = exp_lambda_1

        self.dist0 = pl.Exponential(xmin=xmin, parameters=[self.exp_lambda_0], discrete=discrete)
        self.dist1 = pl.Exponential(xmin=xmin, parameters=[self.exp_lambda_1], discrete=discrete)

    def generate_random(self, size):

        probs = np.random.uniform(size=size)
        dist_id = np.zeros(size)
        dist_id[probs > self.p_1] = 1

        random_vars = np.zeros(size)
        s_0s = len(dist_id[dist_id == 0])
        s_1s = size - s_0s

        random_vars[dist_id == 0] = self.dist0.generate_random(s_0s)
        random_vars[dist_id == 1] = self.dist1.generate_random(s_1s)

        return random_vars
