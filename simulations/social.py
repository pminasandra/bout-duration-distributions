# Pranav Minasandra
# pminasandra.github.io
# 26 Jul, 2023

import sys

import numpy as np
import pandas as pd

import survival

from simulations.agentpool import AgentPool
from simulations.agentpoolutils import recs_as_pd_dataframes

def lin_between(p_begin, p_end, num):
    p_begin = float(p_begin)
    p_end = float(p_end)

    def _f(x):
        assert 0 <= x
        assert x <= num

        return (p_end - p_begin)*x/num + p_begin
    return _f


def log_between(p_begin, p_end, num):
    p_begin = float(p_begin)
    p_end = float(p_end)

    l1 = np.log(p_begin)
    l2 = np.log(p_end)

    def _f(x):
        assert 0 <= x
        assert x <= num

        return np.exp((l2 - l1)*(x/num) + l1)
    return _f


def social_sync_simulation():
    pfunc = log_between(1e-1, 1e-2, 10)
    agentpool = AgentPool(10, pfunc)
    agentpool.run(50000)

    for df in recs_as_pd_dataframes(agentpool.data):
        for state in ["A", "B"]:
            hz_rate_table = survival.compute_behavioural_inertia(df, "meerkat", state, hazard_rate=True) #since meerkat is 'default' for now
            print(hz_rate_table)

if __name__ == "__main__":
    pfunc = log_between(1e-2, 1e-4, 10)
    agentpool = AgentPool(10, pfunc)
    agentpool.run(100)
    print(agentpool.data)

