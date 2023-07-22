# Pranav Minasandra
# pminasandra.github.io
# 20 Jul 2023

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
        # TODO: Handle the bottom statement better based on whether it's iter of iters or iter of funcs
        self.prob_switching = prob_switching

        if init_condition is not None:
            self.records = init_condition
        else:
            self.records = np.array([1]*num_agents)

    @property
    def data(self):
        return pd.DataFrame(self.records)
