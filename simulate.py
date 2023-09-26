# Pranav Minasandra
# pminasandra.github.io
# been a while, forgot the day. couple weeks before 24 Sep 2023

import os
import os.path

import matplotlib.pyplot as plt
import numpy as np

import config
import utilities

import simulations
import simulations.parameter_space
import simulations.sconfig
import simulations.social

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

if __name__ != "__main__":
    raise ImportError("simulate.py is not meant to be imported.")

# BLOCK 1: Effect of classification
#simulations.simulate_with_distribution("Exponential")
#simulations.simulate_with_distribution("Power_Law")
#
#fig, ax = plt.subplots()
#
#error_rates = np.linspace(
#    simulations.sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
#    simulations.sconfig.ERRORS_PARAMETER_SPACE_END,
#    simulations.sconfig.ERRORS_PARAMETER_SPACE_NUM
#)/2
#
#heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Exponential_heavy_tails.npout"))
#
#ht_rates = heavy_tails_exp.mean(axis=0)
#print(error_rates)
#print(ht_rates)
#ax.plot(error_rates, ht_rates, label="Exponential")
#ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
#                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)
#
#heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Power_Law_heavy_tails.npout"))
#
#ht_rates = heavy_tails_exp.mean(axis=0)
#ax.plot(error_rates, ht_rates, label="Power Law")
#ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
#                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)
#
#ax.set_xlabel("Classifier error")
#ax.set_ylabel("Proportion of results with heavy-tail best fits")
#
#ax.legend(title="True bout duration distribution")
#utilities.saveimg(fig, "simulation_classification_effect")

simulations.generate_illustration_at_crucial_error()

# BLOCK 2: Mixtures of exponentials
#simulations.check_mixed_exps()


# BLOCK 3: Social reinforcement
#fig, ax = plt.subplots()
#for i in range(10):
#    print(f"Social simulation, iteration {i}")
#    simulations.social.social_sync_simulation(fig, ax)
#
#ax.set_xlabel("Time since start of bout")
#ax.set_ylabel("Hazard rate")
#utilities.saveimg(fig, "social_reinforcement_simulation")
