# Pranav Minasandra
# pminasandra.github.io
# Jan 11, 2022

import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

GRAPHICS = os.path.join(config.FIGURES, "graphics")
os.makedirs(GRAPHICS, exist_ok=True)

def behavioural_inertia_graphic():
    """
    Simple graphic to show the meaning of behavioural inertia
    """

    fig, ax = plt.subplots()

    x = np.arange(0,10,0.1)
    pdf = scipy.stats.norm(0, 3).pdf(x)

    ax.plot(x, pdf, c="black")
    ax.fill_between(x[25:], pdf[25:])
    ax.fill_between(x[35:], pdf[35:])
    ax.set_xticks([2.5,3.5])
    ax.set_xticklabels([r"$t$", r"$t + \epsilon$"])

    ax.set_ylabel("Probability of bout ending")
    ax.set_xlabel("Time")
    utilities.saveimg(fig, "explainer_behavioural_inertia", GRAPHICS)

behavioural_inertia_graphic()
