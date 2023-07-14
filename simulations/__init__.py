# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

import multiprocessing as mp
import os.path

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw as pl
from scipy.stats import norm

import config
import boutparsing
import fitting
import utilities

import simulations.sconfig
import simulations.classifier
import simulations.parameter_space
from simulations.simulator import Simulator

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

cividis = matplotlib.cm.cividis


def _parallel(mean_A, mean_B, bd_distributions, epoch, fit_props):

    error = norm.cdf(mean_A)
    print(error)
    ft_params = {
        'A': (mean_A, simulations.sconfig.FEATURE_DIST_VARIANCE),
        'B': (mean_B, simulations.sconfig.FEATURE_DIST_VARIANCE)
    }

    results = np.array([0] * (len(config.distributions_to_numbers) + 1))
    for i in range(simulations.sconfig.PER_PARAMETER):
        simulator = Simulator(bd_distributions, ft_params, epoch)
        print(error, ":", i)
        simulator.run(1000)
        simulator.records["datetime"] = pd.to_datetime(simulator.records["datetime"], unit='s')
        bayes_predictions = simulations.classifier.bayes_classify(simulator.records)

        true_states = boutparsing.as_bouts(simulator.records, "meerkat")
        predicted_states = boutparsing.as_bouts(bayes_predictions, "meerkat")

#            true_fits = fitting.fits_to_all_states(true_states) #FIXME: This isn't a line of code we need, but it's a line of code we deserve.
        pred_fits = fitting.fits_to_all_states(predicted_states)

        for state in ["A", "B"]:
            fit = pred_fits[state]
            pred_dist_name, pred_dist = fitting.choose_best_distribution(fit, predicted_states[predicted_states["state"] == state]["duration"])
            results[config.distributions_to_numbers[pred_dist_name]] += 1

    results = 0.5*results/simulations.sconfig.PER_PARAMETER
    results[-1] = error
    fit_props.append(list(results))

def simulate_with_power_laws(distribution_name):

    parameter_space = simulations.parameter_space.parameter_values(
        simulations.sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
        simulations.sconfig.ERRORS_PARAMETER_SPACE_END,
        simulations.sconfig.ERRORS_PARAMETER_SPACE_NUM
    )

    if distribution_name == "Power_Law":
        bd_distributions = {
            'A': pl.Power_Law(xmin = config.xmin, parameters=[simulations.sconfig.POWER_LAW_ALPHA], discrete=config.discrete),
            'B': pl.Power_Law(xmin = config.xmin, parameters=[simulations.sconfig.POWER_LAW_ALPHA], discrete=config.discrete)   
        }
    elif distribution_name == "Exponential":
        bd_distributions = {
            'A': pl.Exponential(xmin = config.xmin, parameters=[simulations.sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete),
            'B': pl.Exponential(xmin = config.xmin, parameters=[simulations.sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete)   
        }
    
    epoch = 1.0
    manager = mp.Manager()
    fit_props = manager.list()

    def _gen():
        for x, y in parameter_space:
            yield x, y, bd_distributions, epoch, fit_props

    fig, ax = plt.subplots()

    pool = mp.Pool(10)
    pool.starmap(_parallel, _gen())
    pool.close()

    fit_props = list(fit_props)
    names = list(config.distributions_to_numbers.keys()) + ["errors"]
    fit_props = pd.DataFrame(fit_props, columns=names)
    fit_props.sort_values(by="errors", inplace=True)
    fit_props.to_csv(os.path.join(config.DATA, f"simulation_results_{distribution_name}.csv"), index=False)
    vals = fit_props[list(config.distributions_to_numbers.keys())]
    ax.set_xscale("log")
    ax.stackplot(fit_props["errors"].to_numpy(),
        vals["Exponential"].to_numpy(),
        vals["Lognormal"].to_numpy(),
        vals["Power_Law"].to_numpy(),
        vals["Truncated_Power_Law"].to_numpy(),
        labels = names[:-1]
    )
    ax.set_xlabel("Classifier error")
    ax.set_ylabel("Best fit distribution - proportion")
    print(fit_props)
    ax.legend()
    ax.set_title(f"True distributions: {distribution_name}")
    utilities.saveimg(fig, f"fits-{distribution_name}")
