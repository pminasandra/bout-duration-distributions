# Pranav Minasandra
# pminasandra.github.io
# 20 Jul 2023

from collections.abc import Iterable

import numpy as np
import pandas as pd

from collections.abc import Iterable

# NOTE: Might be a good idea to generalise to different probabilities across behaviours
# This will be an easy fix, and quite useful in the long run.

class AgentPool:
    """
    Simulates a large number of interacting agents to generate 2-behaviour sequences
    Attributes:
        num_agents (int)
        prob_switching (iter or func)
        data (np.ndarray)
    """

    def __init__(self, num_agents, prob_switching, init_condition=None):
        """
        UNDER CONSTRUCTION 
        prob_switching (iter or func)
        init_condition(list-like with len=num_agents or None)
        """ #FIXME

        self.num_agents = num_agents
        if isinstance(prob_switching, Iterable):
            self._probs = self.prob_switching
            assert len(self._probs) == num_agents
            for p in self._probs:
                assert isinstance(p, float)
                assert 0 <= p
                assert p <= 1

            def _prob_switching(self, n):
                assert n <= self.num_agents
                return self._probs[n]

            self.prob_switching = _prob_switching

        elif callable(prob_switching):
            for i in range(self.num_agents):
                p = prob_switching(i)
                assert isinstance(p, float)
                assert 0 <= p
                assert p <= 1
            self.prob_switching = staticmethod(prob_switching)

        else:
            raise TypeError("prob_switching was not appropriate iterable or function")

        if init_condition is not None:
            assert isinstance(init_condition, Iterable)
            for s in init_condition:
                assert s in (-1.0, 1.0)
            self.records = init_condition
        else:
            self.records = np.random.uniform(size=self.num_agents)
            self.records[self.records < 0.5] = -1.0
            self.records[self.records > 0.5] = 1.0

    @property
    def data(self):
        return self.records

    def step(self):
        """
        Steps the AgentPool simulation by one unit of time
        """

        random_draws = np.random.uniform(size=self.num_agents)

        old_recs = self.records.copy()
        if len(old_recs.shape) == 2:
            old_recs = old_recs[-1, :]

        num_state_1 = len(old_recs[old_recs == -1.0])
        num_state_2 = self.num_agents - num_state_1

        prob_1_2 = self.prob_switching(num_state_2)
        prob_2_1 = self.prob_switching(num_state_1)

        mask_state1 = (old_recs == -1.0)
        mask_state2 = ~mask_state1

        old_recs[mask_state1 & (random_draws < prob_1_2)] = 1.0
        old_recs[mask_state2 & (random_draws < prob_2_1)] = -1.0

        self.records = np.vstack((self.records, old_recs))

    def run(self, t):
        for count in range(t):
            self.step()

if __name__ == "__main__":
    def f(n):
        return 0.5

    agentpool = AgentPool(10, f)
    agentpool.run(100)
    print(agentpool.data)
