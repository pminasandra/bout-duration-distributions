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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utilities

from pkgnametbd import simulations
#from pkgnametbd import simulations.parameter_space
#from pkgnametbd import simulations.sconfig
#from pkgnametbd import simulations.social

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

if __name__ != "__main__":
    raise ImportError("simulate.py is not meant to be imported.")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
## BLOCK 1: Effect of classification
# NOTE: Before running the below block, I recommend setting the minimum
# number of needed bouts in config.py to a low number like 50.
# We can afford to be a bit less stringent in the simulations
# And in power-laws, sometimes the number of bouts can dip low.
simulations.simulate_with_distribution("Exponential")
simulations.simulate_with_distribution("Power_Law")

fig, ax = plt.subplots()

error_rates = np.linspace(
    simulations.sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
    simulations.sconfig.ERRORS_PARAMETER_SPACE_END,
    simulations.sconfig.ERRORS_PARAMETER_SPACE_NUM
)/2

# Plots for heavy_tails
heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Exponential_heavy_tails.npout"))

ht_rates = heavy_tails_exp.mean(axis=0)
ax.plot(error_rates, ht_rates, label="Exponential")
ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)

del heavy_tails_exp
heavy_tails_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Power_Law_heavy_tails.npout"))

ht_rates = heavy_tails_exp.mean(axis=0)
ax.plot(error_rates, ht_rates, label="Power Law")
ax.fill_between(error_rates, ht_rates + 0.5*heavy_tails_exp.std(axis=0), 
                    ht_rates - 0.5*heavy_tails_exp.std(axis=0), alpha=0.5)

ax.set_xlabel("Classifier error")
ax.set_ylabel("Proportion of results with heavy-tail best fits")

ax.legend(title="True bout duration distribution")
utilities.saveimg(fig, "simulation_classification_effect_heavy_tails")
plt.cla()

# Plots for lognormal
# Just copied from above
# In a hurry, sorry :(
lognormal_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Exponential_lognormals.npout"))

ht_rates = lognormal_exp.mean(axis=0)
ax.plot(error_rates, ht_rates, label="Exponential", color="purple")
ax.fill_between(error_rates, ht_rates + 0.5*lognormal_exp.std(axis=0), 
                    ht_rates - 0.5*lognormal_exp.std(axis=0), color="purple",
                    alpha=0.5)

lognormal_exp = np.loadtxt(os.path.join(config.DATA, "simulation_Power_Law_lognormals.npout"))

ht_rates = lognormal_exp.mean(axis=0)
ax.plot(error_rates, ht_rates, label="Power Law", color="darkgreen")
ax.fill_between(error_rates, ht_rates + 0.5*lognormal_exp.std(axis=0), 
                    ht_rates - 0.5*lognormal_exp.std(axis=0), color="darkgreen",
                    alpha=0.5)

ax.set_xlabel("Classifier error")
ax.set_ylabel("Proportion of results with lognormal best fits")

ax.legend(title="True bout duration distribution")
utilities.saveimg(fig, "simulation_classification_effect_lognormals")
## BLOCK 2: why strange stuff happens at error = 0.25
simulations.generate_illustration_at_crucial_error()

# BLOCK 3: Mixtures of exponentials
#simulations.check_mixed_exps()

# The above call generates a csv file, which we will now read
#df = pd.read_csv(os.path.join(config.DATA, "mixed_exp_res.csv"))
#ps = df.p.unique()
#exp2s = df.exp2.unique()
#
## Most common fit for each parameter value
#mode_res = []
#for p in ps:
#    for l2 in exp2s:
#        df_sub = df[(df["p"]==p) & (df["exp2"]==l2)]
#        mode_res.append([p, 0.01, l2, df_sub['dist_name'].mode()[0]])
#
#del df
#df_res = pd.DataFrame(mode_res, columns=["p", "exp1", "exp2", "dist_name"])
#fig, ax = plt.subplots()
#
#for dist_name, data in df_res.groupby(df_res.dist_name):
#    ax.scatter(data.p, data.exp1/data.exp2, label=dist_name, marker="s", s=3.0)
#
#ax.set_yscale("log")
#ax.set_xlabel("$p$")
#ax.set_ylabel(r"$\frac{\lambda_1}{\lambda_2}$")
#ax.legend(framealpha=0.6)
#utilities.saveimg(fig, "mixed_exp_res")

# BLOCK 4: Social reinforcement
#plt.cla()
#fig, ax = plt.subplots()
#for i in range(10):
#    print(f"Social simulation, iteration {i}")
#    simulations.social.social_sync_simulation(fig, ax)
#
#ax.set_xlabel("Time since start of bout")
#ax.set_ylabel("Hazard rate")
#utilities.saveimg(fig, "social_reinforcement_simulation")
