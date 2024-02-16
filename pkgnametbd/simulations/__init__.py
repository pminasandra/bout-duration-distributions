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
from . import parameter_space as parameter_space_lib
from .simulator import Simulator
from . import mixed_exponentials


if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

# The below function will be run in parallel many times
def _simulate_and_get_results(sim_count, ft_params, bd_distributions, epoch, fit_results, lognormal_results):

    np.random.seed()
    simulator = Simulator(bd_distributions, ft_params, epoch)
    simulator.run(sconfig.NUM_BOUTS)
    simulator.records["datetime"] = pd.to_datetime(simulator.records["datetime"], unit='s')

    assert simulator.num_features > 1
# because we've decided to try numerous error rates at once

    num_heavy_tails = [0 for param in ft_params]
    num_lognormals = [0 for param in ft_params]

    for i in range(simulator.num_features):
        print(sim_count, ":", i)
        classifications = classifier.bayes_classify(simulator.records[f"feature{i}"])
        recs = simulator.records.copy()
        recs = recs[["datetime", "state"]]
        recs["state"] = classifications

        predicted_bouts = boutparsing.as_bouts(recs, "meerkat") # "meerkat" used only as a stand-in, since the code needs it on the data-processing side but not here
        predicted_bouts = predicted_bouts[predicted_bouts["duration"] >= config.xmin] # What a nasty well-hidden bug! Fixed 24.07.2023
        pred_fits = fitting.fits_to_all_states(predicted_bouts)

        for state in ["A", "B"]:
            fit = pred_fits[state]
            bouts = predicted_bouts[predicted_bouts["state"] == state]
            bouts = bouts["duration"]
            pred_dist_name, pred_dist = fitting.choose_best_distribution(fit, bouts)
            print(i, ":", pred_dist_name)
            if pred_dist_name in ["Power_Law", "Truncated_Power_Law"]:# is it scale-free?
                num_heavy_tails[i] += 1
            elif pred_dist_name == "Lognormal":# is it lognormal?
                num_lognormals[i] += 1

    for nres in range(simulator.num_features):
        num_heavy_tails[nres] /= 2.0
        num_lognormals[nres] /= 2.0

    fit_results.append(num_heavy_tails)
    lognormal_results.append(num_lognormals)


    print("Simulation #", sim_count, "will now exit.")
    return None
    # The mp.Pool() object has absolute garbage garbage collection
    # Deleting all data manually here
    del predicted_bouts
    del pref_fits
    del recs
    del classifications
    del simulator
    del num_lognormals
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
    parameter_space = parameter_space_lib.parameter_values(
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

    fig, ax = plt.subplots()
    _helper_func_for_specific_case(ft_params, bd_distributions, epoch, fig, ax)
    utilities.saveimg(fig, "illustration_max_ht")
    plt.show()

def simulate_with_distribution(distribution_name):
    """
    Simulates numerous datasets from a distribution, passes the, through error-
    prone classifiers, and performs fits on the resulting data.
    """

    parameter_space = parameter_space_lib.parameter_values(
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
    lognormal_results = manager.list()

    ft_params = [{
        "A": (mean_a, sconfig.FEATURE_DIST_VARIANCE),
        "B": (mean_b, sconfig.FEATURE_DIST_VARIANCE)}
        for (mean_a, mean_b) in parameter_space]


    def _gen():
        for sim_count in range(sconfig.PER_PARAMETER):
            yield sim_count, ft_params, bd_distributions, epoch, fit_results, lognormal_results


    pool = mp.Pool(config.NUM_CORES)
    pool.starmap(_simulate_and_get_results, _gen())
    pool.close()
    pool.join()

    fit_results = np.array(fit_results)
    lognormal_results = np.array(lognormal_results)
    np.savetxt(os.path.join(config.DATA, f"simulation_{distribution_name}_heavy_tails.npout"), fit_results)
    np.savetxt(os.path.join(config.DATA, f"simulation_{distribution_name}_lognormals.npout"), lognormal_results)


def _multiprocessing_helper_func(p, expl0, expl1, count, tgtlist, num_sims):
    np.random.seed()
    print(f"Working on p={p}, l1={expl0}, l2={expl1}")
    dist = mixed_exponentials.MixedExponential(p, expl0, expl1)
    vals = dist.generate_random(sconfig.NUM_BOUTS)

    bouts_df = pd.DataFrame({"state":["A"]*sconfig.NUM_BOUTS, "duration":vals})

    pred_fits = fitting.fits_to_all_states(bouts_df)

    for state in ["A"]:
        fit = pred_fits[state]
        bouts = bouts_df[bouts_df["state"] == state]
        bouts = bouts["duration"]
        dist_name, dist = fitting.choose_best_distribution(fit, bouts)
        tgtlist[count] = [p, expl0, expl1, dist_name]


def check_mixed_exps():
    """
    Simulates data from numerous mixed exponentials (with 2 components mixed),
    and checks the best-fit for the resulting data
    """
    import pandas as pd
    ps = np.arange(0.0, 1.0, 0.01)
    per_param_pair = 100
    exp1 = 0.01
    expl2 = 10.0**(np.arange(-6, 1, 1))
    NUM_SIMS = len(ps)*len(expl2)*per_param_pair

    manager = mp.Manager()
    list_ = manager.list()
    list_.extend([0]*NUM_SIMS)

    def parameter_generate():
        i = 0
        for p in ps:
            for exp2 in expl2:
                for j in range(per_param_pair):
                    pars = (p, exp1, exp2, i, list_, NUM_SIMS)
                    yield pars
                    i += 1


    parameter_generator = parameter_generate()
    pool = mp.Pool(config.NUM_CORES)
    pool.starmap(_multiprocessing_helper_func, parameter_generator)
    pool.close()
    pool.join()

    df = pd.DataFrame(list(list_), columns=["p", "exp1", "exp2", "dist_name"])

    df.to_csv(os.path.join(config.DATA, "mixed_exp_res.csv"),
                index=False
             )
