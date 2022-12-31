# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

"""
Implements class Simulator, a utility class to simulate a 2-behaviour sequence for a given amount of time.
"""

import os.path

import pandas as pd
import numpy as np

import config
import utilities

import simulations.sconfig

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

class Simulator:
    """
    Simulates a 2-behaviour behavioural sequence for a given number of bouts for each state.
    """

    def __init__(self, bd_distributions, ft_params, epoch):
        """
        Simulator initialisation:
        Args:
            bd_distributions (dict): where each key is the name of a behavioural state,
                and each value is a bout duration distribution (in epoch units) with a valid generate_random() method.
            ft_params (dict): where each key is the name of a behavioural state,
                and each value is a tuple of mean and standard deviation for a normal distribution.
            epoch (float): the temporal resolution of the behavioural sequences to be generated.
        """
        for state in bd_distributions:
            assert bd_distributions[state].generate_random

        self.bd_distributions = bd_distributions
        self.ft_params = ft_params
        self.epoch = int(epoch)

    def run(self, num_bouts):
        """
        Runs the Simulator, generates toy features
        """

        bout_values = {}
        feature_values = {}
        states = list(self.bd_distributions.keys())
        for state in self.bd_distributions:
            bout_values[state] = np.array(self.bd_distributions[state].generate_random(num_bouts)) * self.epoch


        records = {"datetime": [], "state": [], "feature": []}

        current_time = 0
        for i in range(num_bouts):
            current_state = states[i % 2]
            current_bout = int(bout_values[state][i])
            mean, stdev = self.ft_params[current_state]

            records["state"].extend([current_state] * current_bout)
            records["datetime"].extend(range(int(current_time), int(current_time + current_bout), int(self.epoch)))
            records["feature"].extend(list(np.random.normal(mean, stdev, current_bout)))

            current_time += current_bout

        self.records = pd.DataFrame(records)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import powerlaw as pl

    bd_distributions = {
        'A': pl.Power_Law(xmin = config.xmin, parameters=[2.0], discrete=True),
        'B': pl.Power_Law(xmin = config.xmin, parameters=[2.0], discrete=True)
    }

    ft_params = {
        'A': (-1, 0.005),
        'B': (1, 0.005)
    }

    epoch = 1.0

    simulator = Simulator(bd_distributions, ft_params, epoch)
    simulator.run(1000)

    records = simulator.records
    ft_A = records[records["state"] == "A"]
    ft_B = records[records["state"] == "B"]
    plt.plot(records["feature"], color="black")
    plt.eventplot(ft_A["time"], color="red", alpha=0.3)
    plt.eventplot(ft_B["time"], color="blue", alpha=0.3)
    plt.show()

