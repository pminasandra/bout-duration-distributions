#! /usr/bin/env python 

# Pranav Minasandra
# pminasandra.github.io
# been a while, forgot the day. couple weeks before 24 Sep 2023

"""
Runs a suite of 4 simulations (BLOCKs below)
    (1) How does an error-prone classifier affect distribution fit?
    (2) Why does strange fit behavior occur at error = 0.25?
    (3) If I mix two exponential distributions with random parameters,
        do they happen to look power-law like?
    (4) If animals socially synchronise their behaviors, do they look like
        they self-reinforce their bouts (i.e., have decreasing hazard functions?
"""

import os
import os.path

import matplotlib.pyplot as plt
import numpy as np

import config
import utilities

from pkgnametbd import simulations
from pkgnametbd import simulations.parameter_space
from pkgnametbd import simulations.sconfig
from pkgnametbd import simulations.social

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

if __name__ != "__main__":
    raise ImportError("simulate.py is not meant to be imported.")

# BLOCK 1: Effect of classification
simulations.simulate_with_distribution("Exponential")
simulations.simulate_with_distribution("Power_Law")

fig, ax = plt.subplots()

# > BROCK OPT
# Probably this should go above the simulations and get
# passed throught to parameter_space.py to ensure
# that axis settings are in sync with how data was generated
# <
error_rates = np.linspace(
    simulations.sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
    simulations.sconfig.ERRORS_PARAMETER_SPACE_END,
    simulations.sconfig.ERRORS_PARAMETER_SPACE_NUM
)/2

# > BROCK OPT
# could be more DRY...
# for dist in ["Exponential", "Power Law"]:
# <

heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Exponential_heavy_tails.npout"))

ht_rates = heavy_tails_exp.mean(axis=0) #BROCK TODO: doublecheck this
print(error_rates)
print(ht_rates)
ax.plot(error_rates, ht_rates, label="Exponential")
ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)

heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Power_Law_heavy_tails.npout"))

ht_rates = heavy_tails_exp.mean(axis=0)
ax.plot(error_rates, ht_rates, label="Power Law")
ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)

ax.set_xlabel("Classifier error")
ax.set_ylabel("Proportion of results with heavy-tail best fits")

ax.legend(title="True bout duration distribution")

# BROCK REQ >
# It doesn't seem like you are plotting or outputting 
# anywhere a summary of the findings re: scale-free rate,
# but I think that is referenced in the paper.
# <
utilities.saveimg(fig, "simulation_classification_effect")

# BLOCK 2: why strange stuff happens at error = 0.25
simulations.generate_illustration_at_crucial_error()

# BLOCK 3: Mixtures of exponentials
simulations.check_mixed_exps()


# BLOCK 4: Social reinforcement
fig, ax = plt.subplots()
for i in range(10):
    print(f"Social simulation, iteration {i}")
    simulations.social.social_sync_simulation(fig, ax)

ax.set_xlabel("Time since start of bout")
ax.set_ylabel("Hazard rate")
utilities.saveimg(fig, "social_reinforcement_simulation")
