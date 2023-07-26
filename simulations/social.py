# Pranav Minasandra
# pminasandra.github.io
# 26 Jul, 2023

import numpy as np
import pandas as pd

from simulations.agentpool import AgentPool

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

if __name__ == "__main__":
    pfunc = log_between(1e-2, 1e-4, 10)
    agentpool = AgentPool(10, pfunc)
    agentpool.run(100)
    print(agentpool.data)

