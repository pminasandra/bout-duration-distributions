# Pranav Minasandra
# pminasandra.github.io
# 20 Jul 2023

from collections.abc import Iterable

import numpy as np
import pandas as pd

class AgentPool:
    """
    Simulates a large number of interacting agents to generate 2-behaviour sequences
    Attributes:
        num_agents (int)
        prob_switching (iter of iter or func)

    """

    def __init__(self, num_agents, prob_switching, init_condition=None):
        """
        UNDER CONSTRUCTION 
        """ #FIXME

        self.num_agents = num_agents
        assert (isinstance(prob_switching, Iterable))
        self.prob_switching = []
        for item in prob_switching:
            if isinstance(item, Iterable):
                itemc = np.array(item)
                assert itemc.dtype == float
                assert len(itemc) == num_agents
            elif callable(item):
                itemc = item
            else:
                raise TypeError("AgentPool() __init__: Expected iterable or callable")
            self.prob_switching.append(itemc)

        if init_condition is not None:
            self.records = init_condition
        else:
            self.records = np.array([1]*num_agents)

    @property
    def data(self):
        return pd.DataFrame(self.records)

    def step(self)
