# Pranav Minasandra
# pminasandra.github.io
# 26 Jul, 2023

"""
Wrapper around agentpool and agentpoolutils to perform social interaction
+ behavior dynamics sims
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import survival

from simulations.agentpool import AgentPool
from simulations.agentpoolutils import recs_as_pd_dataframes

def _lin_between(p_begin, p_end, num):
    p_begin = float(p_begin)
    p_end = float(p_end)

    def _f(x):
        assert 0 <= x
        assert x <= num

        return (p_end - p_begin)*x/num + p_begin
    return _f


def _log_between(p_begin, p_end, num):
    p_begin = float(p_begin)
    p_end = float(p_end)

    l1 = np.log(p_begin)
    l2 = np.log(p_end)

    def _f(x):
        assert 0 <= x
        assert x <= num

        return np.exp((l2 - l1)*(x/num) + l1)
    return _f


def social_sync_simulation(num_ind):
    """
    Simulates social pool of interacting agents, plots hazard 
    functions of their bouts
    """
    pfunc = _log_between(1e-2, 1e-1, num_ind)
    agentpool = AgentPool(num_ind, pfunc)
    agentpool.run(50000)

    hz_tables = []
    for df in recs_as_pd_dataframes(agentpool.data):
        for state in ["A", "B"]: #WLOG since A and B are identical here
            hz_rate_table = survival.compute_behavioural_inertia(df, "meerkat",
                            state, hazard_rate=True) #since meerkat is 'default' for now
            hz_tables.append(hz_rate_table)
            lenmin = min([arr.shape[0] for arr in hz_tables])
            lenmin = min(50, lenmin)
            hz_tables = [arr[0:lenmin,] for arr in hz_tables]

    return hz_tables

if __name__ == "__main__":
    num_ind = 10
    pfunc = _log_between(1e-2, 1e-4, 10)
    print(f"Running social sync trials for {num_ind} individuals.")
    agentpool = AgentPool(num_ind, pfunc)
    agentpool.run(100)
    print(agentpool.data)

