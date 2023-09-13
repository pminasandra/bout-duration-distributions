# Pranav Minasandra
# pminasandra.github.io
# 26 Jul, 2023

import numpy as np
import pandas as pd

import boutparsing

def recs_as_pd_dataframes(data):
    """
    Helper function, GENERATOR.
    Yields pd.DataFrame objects containing bout sequences of an individual each.
    Args:
        data (np.ndarray): typically AgentPool().data
    Yields:
        pd.DataFrame
    """

    assert len(data.shape) > 1, "Did you forget to AgentPool(...).run(...)?"
    assert data.shape[0] > 1

    time, num_agents = data.shape

    for i in range(num_agents):
        state = data[:, i]
        state_copy = np.empty(state.shape, dtype=str)
        
        state_copy[state == -1.0] = "A"
        state_copy[state == 1.0] = "B"
        datetime = list(range(time))

        df = pd.DataFrame({"datetime": datetime, "state": state_copy}) 
        df["datetime"] = pd.to_datetime(df["datetime"], unit='s')

        yield boutparsing.as_bouts(df, "meerkat") #since meerkat is our 'default' for now

