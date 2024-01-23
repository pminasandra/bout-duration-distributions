# Pranav Minasandra
# pminasandra.github.io
# Jan 11, 2022

import glob
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pkgnametbd import config
from pkgnametbd import utilities

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
    utilities.saveimg(fig, "explainer_behavioural_inertia", directory=GRAPHICS)

def hazard_and_ccdf():
    """
    Graphic showing burn-in, wear-out, and memorilessness
    """

    fig, axs = plt.subplots(3,2, sharey=True, tight_layout=True)
    exp = scipy.stats.expon(scale=3)   # Memoriless
    norm = scipy.stats.norm(scale=5.3) # Wear-out
    pl = scipy.stats.pareto(b=1.5)       # Burn-in

    x = np.arange(1, 13, 0.1)
    axs[0,0].plot(x, exp.pdf(x)/exp.sf(x))
    axs[0,0].set_title("Memoriless")

    axs[1,0].plot(x, norm.pdf(x)/norm.sf(x))
    axs[1,0].set_title("Wear-out")
    axs[1,0].set_ylabel("Hazard rate")

    axs[2,0].plot(x, pl.pdf(x)/pl.sf(x))
    axs[2,0].set_title("Burn-in")
    axs[2,0].set_xlabel("Time")

    axs[0,1].plot(x, exp.sf(x))
    axs[0,1].set_title("Memoriless")

    axs[1,1].plot(x, norm.sf(x))
    axs[1,1].set_title("Wear-out")
    axs[1,1].set_ylabel("CCDF")

    axs[2,1].plot(x, pl.sf(x))
    axs[2,1].set_title("Burn-in")
    axs[2,1].set_xlabel("Time")

    for ax in axs:
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")

    utilities.saveimg(fig, "explainer_three_types", directory=GRAPHICS)

def behavioural_sequence():
    ts = []
    states = []

    for i in range(1000):
        ts.append(i)
        if random.random() < 0.5:
            states.append(0)
        else:
            states.append(1)

    import pandas as pd
    df = pd.DataFrame({"time":ts, "state":states})
    df0 = df[df["state"] == 0]
    df1 = df[df["state"] == 1]

    fig, ax = plt.subplots()
    ax.eventplot(df1["time"])
    ax.eventplot(df0["time"], color="red")

    utilities.saveimg(fig, "decor_behavioural_sequence", directory=GRAPHICS)

def memory_and_memoriless():

    from matplotlib.animation import FuncAnimation, FFMpegWriter

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    ax_memoriless_race = axs[0, 0]
    ax_memoriless_hist = axs[0, 1]
    ax_memory_race = axs[1, 0]
    ax_memory_hist = axs[1, 1]


    N = 10000
    T = 10000
    p_memoriless = 0.001
    b_memory = 1.5

    animation_resolution = 100 # Once every so many cycles

    # initialise N racers in each case
    memoriless_racers = np.array([1]*N)
    memoriless_racers_stats = np.array([1]*N)
    memory_racers = np.array([1]*N)
    memory_racers_stats = np.array([1]*N)

    # Run simulation
    count = 0
    def animate(count):
        if count >= T:
            return

        for axis in fig.axes:
            axis.clear()

        ax_memoriless_race.xaxis.set_ticks([], [])
        ax_memory_race.xaxis.set_ticks([], [])

        ax_memoriless_race.set_ylim([0, 10000])
        ax_memory_race.set_ylim([0, 10000])
        ax_memoriless_race.set_xlim([0, 10000])
        ax_memory_race.set_xlim([0, 10000])
        ax_memoriless_race.set_title("Memoriless runners")
        ax_memory_race.set_title("Runners with memory")
        ax_memoriless_hist.set_title("Frequency of distances run")
        ax_memory_hist.set_title("Frequency of distances run")

        for i in range(animation_resolution):
            memoriless_racers[memoriless_racers_stats == 1] += 1
            memory_racers[memory_racers_stats == 1] += 1
            prob_survivals = np.random.uniform(size=N) * memoriless_racers_stats
            memoriless_racers_stats[prob_survivals < p_memoriless] = 0
            prob_survivals = np.random.uniform(size=N) * memory_racers_stats
            memory_racers_stats[prob_survivals < (b_memory - 1)*np.log(1 + 1/memory_racers)] = 0

        memoriless_survivors = (memoriless_racers_stats == 1)
        memory_survivors = (memory_racers_stats == 1)

        ax_memoriless_hist.hist(memoriless_racers, 200)
        ax_memory_hist.hist(memory_racers, 200)
        ax_memoriless_race.scatter(np.arange(N)[memoriless_survivors], memoriless_racers[memoriless_survivors], color="#00aa00", s=1.2)
        ax_memoriless_race.scatter(np.arange(N)[~memoriless_survivors], memoriless_racers[~memoriless_survivors], color="black", s=0.12)
        ax_memory_race.scatter(np.arange(N)[memory_survivors], memory_racers[memory_survivors], color="#00aa00", s=1.2)
        ax_memory_race.scatter(np.arange(N)[~memory_survivors], memory_racers[~memory_survivors], color="black", s=0.12)

    anim = FuncAnimation(fig, animate, frames=np.arange(0, T, animation_resolution), interval=20, repeat=False)
    
    writer = FFMpegWriter(metadata={"title":"Memorilessness vs Self-Reinforcement",
                                                "artist":"Pranav \"Baba\" Minasandra", 
                                                "year":"2023", 
                                                "author":"Pranav \"Baba\" Minasandra"})
    os.makedirs(os.path.join(GRAPHICS, "animations"), exist_ok=True)
    anim.save(os.path.join(GRAPHICS, "animations", "explainer_memory.gif"), writer=writer)


def make_pie_charts():
    """
    Makes pie charts showing proportion of fits best matching powerlaw or truncated powerlaw
    """

    fitresults = os.path.join(config.DATA, "FitResults")
    for species in config.species:
        reldatadir = os.path.join(fitresults, species)

        for datafile in glob.glob(os.path.join(reldatadir, "*.csv")):
            if species == 'meerkat' and 'Running' in datafile:
                continue

            df = pd.read_csv(datafile, sep=',')
            tot_inds = df.shape[0]

            tot_heavy = (df['Power_Law'] == 0).sum()\
                        + (df['Truncated_Power_Law'] == 0).sum()
            fig, ax = plt.subplots()

            prop = tot_heavy/tot_inds
            ax.pie([prop, 1-prop], colors=['maroon', 'gray'])

            state = os.path.basename(datafile)[:-len(".csv")]
            utilities.saveimg(fig, f"piechart-{species}-{state}")

if __name__ == "__main__":
    # behavioural_inertia_graphic()
    # hazard_and_ccdf()
    # behavioural_sequence()
    #memory_and_memoriless()
    make_pie_charts()
