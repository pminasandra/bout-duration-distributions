# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

"""
Implements class Simulator, a utility class to simulate a 2-behaviour sequence for a given amount of time.
"""

import os.path

import pandas as pd
import numpy as np

from pkgnametbd import config
from pkgnametbd import utilities

from . import sconfig

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint
    
# BROCK OPT >
# Since this object has only one method and you are only running that 
# method once per instantiation, making this an object/class doens't seem to
# be adding much over a simple function that returns records.
#
# In fact, it's a bit counterintuitive (to me at least) that run() merely sets
# s.records and does not return anything.
# <

class Simulator:
    """
    Simulates a 2-behaviour behavioural sequence for a given number of bouts for each state.
    Example usage:
    >>> s = Simulator(...)
    >>> s.run(...)
    >>> s.records
    <pd.DataFrame> with keys "datetime", "state", and "feature"
    """
    # BROCK OPT >
    # The term "feature" here is a bit confusing to me. Maybe it's a term in the literature
    # That I'm not too familliar with? When I think of "feature" I think of a predictor
    # variable in an ML model. It seems like here "state" is true state and each "feature"
    # is a measurement where the state is an input to the distribution of that measurement.
    # <
    def __init__(self, bd_distributions, ft_params, epoch):
        """
        Initialises simulator

        Args:
            bd_distributions (dict): where each key is the name of a behavioural state,
                and each value is a bout duration distribution (in epoch units) with a valid generate_random() method.
            ft_params (dict or iterable of dicts): where each key is the name of a behavioural state,
                and each value is a tuple of mean and standard deviation for a normal distribution.
            epoch (float): the temporal resolution of the behavioural sequences to be generated.
        """

        # First, check if multiple features are needed
        self.multiple_features = False
        self.num_features = 1
        assert type(ft_params) in [dict, list]
        if type(ft_params) == list:
            for ft_needed in ft_params:
                assert type(ft_needed) == dict
            self.multiple_features = True
            # BROCK OPT >
            # This attribute is a bit of a misnomer in that
            # if you a list with one element
            # then multiple features will be set to true when, in fact
            # there is only one feature.
            #
            # It appears that both cases are still handled correctly.
            # Seems to add unecessary complexity though to accept 
            # either a list of objects or the object itself.
            #
            # Why not just force the caller to put their one element inside a list?
            # <
            self.num_features = len(ft_params)

        # BROCK OPT >
        # Since you only handle two states, you may want to assert
        # len(bd_distributions) = True
        for state in bd_distributions:
            assert bd_distributions[state].generate_random

        # Proceeding as usual
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


        if not self.multiple_features:
            records = {"datetime": [], "state": [], "feature": []}
        else:
            records = {"datetime": [], "state": []}
            for i in range(len(self.ft_params)):
                records[f"feature{i}"] = []


        current_time = 0
        i = 0
        while i < num_bouts and current_time < sconfig.MAX_REC_TIME:
            current_state = states[i % 2]
            # BROCK OPT > 
            # It looks like you are generating twice as many bout values
            # as you are using. If I'm understanding correctly, you're
            # only using odd indexed bout values from one state and even
            # indexed bout values from the other. Generating random 
            # values can sometimes be one of the more compute 
            # intensive parts of simulations so probably not great to do
            # so needlessly. (If that's what's going on.)
            # <
            current_bout = int(bout_values[current_state][i])
            if current_time + current_bout > sconfig.MAX_REC_TIME:
                current_bout = sconfig.MAX_REC_TIME - current_time

            if not self.multiple_features:
                mean, stdev = self.ft_params[current_state]

            records["state"].extend([current_state] * current_bout)
            records["datetime"].extend(range(int(current_time), int(current_time + current_bout), int(self.epoch)))
            if not self.multiple_features:
                records["feature"].extend(list(np.random.normal(mean, stdev, current_bout)))

            if self.multiple_features:
                for j in range(len(self.ft_params)):
                    mean, stdev = self.ft_params[j][current_state]
                    records[f"feature{j}"].extend(list(np.random.normal(mean, stdev, current_bout)))

            current_time += current_bout
            i += 1

        self.records = pd.DataFrame(records)

if __name__ == "__main__":
# The below code is just an example provided to generate a helpful illustration
    import matplotlib.pyplot as plt
    import powerlaw as pl

    bd_distributions = {
        'A': pl.Power_Law(xmin = config.xmin, parameters=[2.0], discrete=True),
        'B': pl.Power_Law(xmin = config.xmin, parameters=[2.0], discrete=True)
    }

    ft_params = [{
        'A': (-1, 0.75),
        'B': (1, 0.75)
    }, {
        'A': (-1, 0.05),
        'B': (1, 0.05)
    }]

    epoch = 1.0

    simulator = Simulator(bd_distributions, ft_params, epoch)
    simulator.run(50)

    records = simulator.records
    ft_A = records[records["state"] == "A"]
    ft_B = records[records["state"] == "B"]
    #plt.plot(records["feature1"], color="gray", linewidth=0.3)
    plt.eventplot(ft_A["datetime"], lineoffsets=2.5, linelengths=0.25, color="red", alpha=0.3)
    plt.eventplot(ft_B["datetime"], lineoffsets=2.5, linelengths=0.25, color="blue", alpha=0.3)

    import classifier

    classifications = classifier.bayes_classify(simulator.records["feature0"])
    df2 = pd.DataFrame({"datetime":records.datetime, "state":classifications})
    ft_A = df2[classifications == "A"]
    ft_B = df2[classifications == "B"]
    #plt.plot(classifications["feature1"], color="gray", linewidth=0.3)
    plt.eventplot(ft_A["datetime"], lineoffsets=-2.5, linelengths=0.25, color="red", alpha=0.3)
    plt.eventplot(ft_B["datetime"], lineoffsets=-2.5, linelengths=0.25, color="blue", alpha=0.3)

    plt.plot(records["feature0"], color="black", linewidth=0.3)
    plt.xlabel("Time")
    plt.ylabel("Feature value")

    plt.savefig(os.path.join(config.PROJECTROOT, "temp", "illustration_classification_effect.pdf"))
    plt.savefig(os.path.join(config.PROJECTROOT, "temp", "illustration_classification_effect.png"))

