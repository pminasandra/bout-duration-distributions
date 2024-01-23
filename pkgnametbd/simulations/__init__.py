# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

import multiprocessing as mp
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw as pl

from pkgnametbd import config
from pkgnametbd import boutparsing
from pkgnametbd import fitting
from pkgnametbd import utilities

from . import sconfig
from . import classifier
from . import parameter_space
from .simulator import Simulator
from . import mixed_exponentials

# > BROCK OPT
# Why not use a logger?
# < 
if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

# > BROCK OPT
# Seems like the arg "sim_count" is more like sim id or sim index
# count sounds like a total or a sum
# <
# The below function will be run in parallel many times
def _simulate_and_get_results(sim_count, ft_params, bd_distributions, epoch, fit_results, fit_results_spec):

    np.random.seed()
    simulator = Simulator(bd_distributions, ft_params, epoch)
    simulator.run(sconfig.NUM_BOUTS)
    simulator.records["datetime"] = pd.to_datetime(simulator.records["datetime"], unit='s')

    assert simulator.num_features > 1
# because we've decided to try numerous error rates at once

    num_heavy_tails = [0 for param in ft_params]
    num_heavy_tails_param_range = [0 for param in ft_params]

    for i in range(simulator.num_features):
        print(sim_count, ":", i)
        classifications = classifier.bayes_classify(simulator.records[f"feature{i}"])
        # BROCK OPT >
        # It looks like you are copying a datatable here and then throwing away 
        # all except one column of it. There is probably a more efficient thing to do here.
        # <
        recs = simulator.records.copy()
        recs = recs[["datetime", "state"]]
        recs["state"] = classifications

        # BROCK OPT >
        # Looks like this is a placeholder species. 
        # maybe call it "placeholder" or "simulated species"
        # to make it clear that it's just a label and not impacting
        # the logic inside the function.
        # In the line after...what was the bug? I'm very curious
        # <
        predicted_bouts = boutparsing.as_bouts(recs, "meerkat") # "meerkat" used only as a stand-in, since the code needs it on the data-processing side but not here
        predicted_bouts = predicted_bouts[predicted_bouts["duration"] >= config.xmin] # What a nasty well-hidden bug! Fixed 24.07.2023
        pred_fits = fitting.fits_to_all_states(predicted_bouts)

        for state in ["A", "B"]:
            fit = pred_fits[state]
            bouts = predicted_bouts[predicted_bouts["state"] == state]
            bouts = bouts["duration"]
            pred_dist_name, pred_dist = fitting.choose_best_distribution(fit, bouts)
            # BROCK OPT >
            # Why not save the actual predicted dist and params for each sim? (e.g. as pickle)
            # you're only doing 200 iterations, so not that much storage to save more metadata
            # even if you don't want to save entire input dataset
            # <
            if pred_dist_name in ["Power_Law", "Truncated_Power_Law"]:# is it scale-free?
                num_heavy_tails[i] += 1
                if pred_dist.alpha < 2.0:# is it scale-free with fun params?
                    num_heavy_tails_param_range[i] += 1
    # BROCK REQ >
    # Why are we dividing by 2 here? can you at least add a comment to explain?
    # <

    for nres in range(simulator.num_features):
        num_heavy_tails[nres] /= 2.0
        num_heavy_tails_param_range[nres] /= 2.0

    fit_results.append(num_heavy_tails)
    fit_results_spec.append(num_heavy_tails_param_range)


    print("Simulation #", sim_count, "will now exit.")
    return None
    # BROCK OPT >
    # I don't think the below code actually runs because it's after
    # the return statement.
    # <
    # The mp.Pool() object has absolute garbage garbage collection
    # Deleting all data manually here
    del predicted_bouts
    del pref_fits
    del recs
    del classifications
    del simulator
    del num_heavy_tails_param_range
    del num_heavy_tails


def _helper_func_for_specific_case(ft_params, bd_distributions, epoch, fig, ax):
    # smelly code, but this just generates an illustration for a figure in the appendix.
    # not really generally applicable
    for i in range(20):
        print(i)
        np.random.seed()
        simulator = Simulator(bd_distributions, ft_params, epoch)
        simulator.run(sconfig.NUM_BOUTS)
        simulator.records["datetime"] = pd.to_datetime(simulator.records["datetime"], unit='s')

        actual_bouts = boutparsing.as_bouts(simulator.records[["datetime", "state"]], "meerkat")
        actual_bouts = actual_bouts[actual_bouts["duration"] >= config.xmin] 

        classifications = classifier.bayes_classify(simulator.records[f"feature5"])
        recs = simulator.records.copy()
        recs = recs[["datetime", "state"]]
        recs["state"] = classifications

        predicted_bouts = boutparsing.as_bouts(recs, "meerkat") # "meerkat" used only as a stand-in, since the code needs it on the data-processing side but not here
        predicted_bouts = predicted_bouts[predicted_bouts["duration"] >= config.xmin] # What a nasty well-hidden bug! Fixed 24.07.2023

        for state in ["A", "B"]:
            act_durs = actual_bouts[actual_bouts["state"] == state].duration
            pred_durs = predicted_bouts[predicted_bouts["state"] == state].duration

            fit_actual = pl.Fit(act_durs, xmin=config.xmin, discrete=config.discrete)
            fit_predicted = pl.Fit(pred_durs, xmin=config.xmin, discrete=config.discrete)

            if i == 0 and state=="A":
                act_label = "True bouts"
                pred_label = "Bouts after classification"
            elif i > 0 or state == "B":
                act_label = None
                pred_label = None
            fit_actual.plot_ccdf(ax=ax, color="maroon", linewidth=0.5, label=act_label)
            fit_predicted.plot_ccdf(ax=ax, color="maroon", linewidth=0.5, linestyle="dotted", label=pred_label)
    ax.legend()
    ax.set_xlabel("Bout duration (t)")
    ax.set_ylabel(r"Pr$(\tau > t)$")


def generate_illustration_at_crucial_error():
    """
    This generates an image with error rate 0.277.
    This is useful for a figure in our paper, but is not generally
    applicable at this stage.
    """
    # > BROCK OPT
    # A lot of the code here is duplciated with the function below.
    # Could they not call a shared function to avoid this?
    # <
    # > BROCK OPT
    # Here you are giving your variable the same name as the module.
    # This is bad practice because it prevents you from being able to use
    # that module anymore, and it creates confusion.
    # <
    parameter_space = parameter_space.parameter_values(
        sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
        sconfig.ERRORS_PARAMETER_SPACE_END,
        sconfig.ERRORS_PARAMETER_SPACE_NUM
    )

    bd_distributions = {
        'A': pl.Exponential(xmin = config.xmin, parameters=[sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete),
        'B': pl.Exponential(xmin = config.xmin, parameters=[sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete)   
    }

    epoch = 1.0
    ft_params = [{
        "A": (mean_a, sconfig.FEATURE_DIST_VARIANCE),
        "B": (mean_b, sconfig.FEATURE_DIST_VARIANCE)}
        for (mean_a, mean_b) in parameter_space]

    vals = ft_params[5] # spurious detection of exponential as tpl
    
    # > BROCK OPT
    # You're using two different patterns with this and the function below.
    # In the main function (below), you have the simulations module write the
    # data and simulate.py plot it.
    # 
    # Here the simulation module itself is doing the plotting.
    # 
    # My preferred pattern would be to have one module that generates the
    # data  (this simulate module), a seperate one that takes in the data and 
    # generates the plot (maybe a plot_sim_results module).
    # Then the launch script (simulate.py) calls these two in sequence and 
    # the data to be plotted is passed as python objects instead of being 
    # written and read. It's a summary, no? so small enough to be passed in this way.
    # (Maybe write to disk as a backup/fallback if you're afraid of interuptions.)
    #
    # <

    fig, ax = plt.subplots()
    _helper_func_for_specific_case(ft_params, bd_distributions, epoch, fig, ax)
    utilities.saveimg(fig, "illustration_max_ht")
    plt.show()

def simulate_with_distribution(distribution_name):
    """
    Simulates numerous datasets from a distribution, passes the, through error-
    prone classifiers, and performs fits on the resulting data.
    """

    # > BROCK OPT
    # Consider passing the configs into the function in some form or fashion.
    # <
    parameter_space = parameter_space.parameter_values(
        sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
        sconfig.ERRORS_PARAMETER_SPACE_END,
        sconfig.ERRORS_PARAMETER_SPACE_NUM
    )

    
    if distribution_name == "Power_Law":
        bd_distributions = {
            'A': pl.Power_Law(xmin = config.xmin, parameters=[sconfig.POWER_LAW_ALPHA], discrete=config.discrete),
            'B': pl.Power_Law(xmin = config.xmin, parameters=[sconfig.POWER_LAW_ALPHA], discrete=config.discrete)   
        }

    elif distribution_name == "Exponential":
        bd_distributions = {
            'A': pl.Exponential(xmin = config.xmin, parameters=[sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete),
            'B': pl.Exponential(xmin = config.xmin, parameters=[sconfig.EXPONENTIAL_LAMBDA], discrete=config.discrete)   
        }
    
    epoch = 1.0
    manager = mp.Manager()
    fit_results = manager.list()
    fit_results_spec = manager.list()

    ft_params = [{
        "A": (mean_a, sconfig.FEATURE_DIST_VARIANCE),
        "B": (mean_b, sconfig.FEATURE_DIST_VARIANCE)}
        for (mean_a, mean_b) in parameter_space]


    def _gen():
        for sim_count in range(sconfig.PER_PARAMETER):
            yield sim_count, ft_params, bd_distributions, epoch, fit_results, fit_results_spec


    pool = mp.Pool(config.NUM_CORES)
    pool.starmap(_simulate_and_get_results, _gen())
    pool.close()
    pool.join()

    fit_results = np.array(fit_results)
    fit_results_spec = np.array(fit_results_spec)
    np.savetxt(os.path.join(config.DATA, f"simulation_{distribution_name}_heavy_tails.npout"), fit_results)
    np.savetxt(os.path.join(config.DATA, f"simulation_{distribution_name}_heavy_tails_params.npout"), fit_results_spec)


def _multiprocessing_helper_func(p, expl0, expl1, count, tgtlist, num_sims):
    np.random.seed()
    dist = mixed_exponentials.MixedExponential(p, expl0, expl1)
    vals = dist.generate_random(sconfig.NUM_BOUTS)

    bouts_df = pd.DataFrame({"state":["A"]*sconfig.NUM_BOUTS, "duration":vals})

    pred_fits = fitting.fits_to_all_states(bouts_df)

    for state in ["A"]:
        fit = pred_fits[state]
        bouts = bouts_df[bouts_df["state"] == state]
        bouts = bouts["duration"]
        dist_name, dist = fitting.choose_best_distribution(fit, bouts)
        tgtlist[count] = dist_name


def check_mixed_exps():
    """
    Simulates data from numerous mixed exponentials (with 2 components mixed),
    and checks the best-fit for the resulting data
    """
    import pandas
    NUM_SIMS = 5000
    expl1 = 0.01

    manager = mp.Manager()
    list_ = manager.list()
    list_.extend([0]*NUM_SIMS)

    def parameter_generate():
        exp2s = 10**(-3*np.random.uniform(size=NUM_SIMS) - 1)
        ps = np.random.uniform(size=NUM_SIMS)
        for i in range(NUM_SIMS):
            pars = (ps[i], expl1, exp2s[i], i, list_, NUM_SIMS)
            yield pars


    parameter_generator = parameter_generate()
    pool = mp.Pool(config.NUM_CORES - 3)
    pool.starmap(_multiprocessing_helper_func, parameter_generator)
    pool.close()
    pool.join()

    with open(os.path.join(config.DATA, "mixed_exp_fits.txt"), "w") as fd:
        for dist in list_:
            old_print(dist, file=fd)
