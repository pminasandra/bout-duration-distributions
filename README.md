--- 
title: Bout duration distributions in animals 
author: Pranav Minasandra 
---

# Analysing behavioural dynamics in the wild.
![a montage of animal behaviours](./assets/montage.png)

# Details
The analyses in this code forms the basis of our upcoming paper in PNAS
[(pre-print)](https://www.biorxiv.org/content/10.1101/2024.01.20.576411v3):
'Behavioral sequences across multiple animal species in the wild share common structural features'. 

## People who worked actively on this code
[Pranav Minasandra](https://pminasandra.github.io)

## People who performed a code review
Katrina Brock and Ariana Strandburg-Peshkin

## People whose contributions were necessary for this project to get going
[Ariana Strandburg-Peshkin](https://cocomo.group),
Emily Grout,
Katrina Brock,
[Meg Crofoot](https://www.ab.mpg.de/crofoot),
Vlad Demartsev,
[Andy Gersick](https://circle-polygon-6hmw.squarespace.com/),
Ben Hirsch,
[Kay Holekamp](https://www.holekamplab.org/),
[Lily Johnson-Ulrich](http://lilyjohnsonulrich.weebly.com/),
Amlan Nayak,
Josue Ortega,
[Marie Roch](https://roch.sdsu.edu/),
[Eli Strauss](https://straussed.github.io/),
Marta Manser,
Frants Jensen,
Baptiste Averly,
and many others

# Overview

This project ties together results from behavioural classifiers built using
[hyenas](https://github.com/pminasandra/hyena-acc),
[meerkats](https://github.com/pminasandra/meerkat-acc),
[coatis](https://github.com/pminasandra/Coati_ACC_Pipeline). Here, I find
long timescale patterns in behavioural dynamics for all classified behaviours for each individual of
each species. This project stems from my serendipitous discovery of
heavy-tailed bout duration distributions in spotted hyenas in 2019.

Heavy-tailed distributions of bout durations could imply that self-reinforcement
plays a role in behavioural dynamics at the fine scale: such distributions have
decreasing associated hazard rates, which means that the longer the animal is in
a behavioural state, the less likely it becomes to exit that behaviour in the
next instant. This discovery implies that wildly
different mammals have decreasing hazard rates for all behavioural states.
We also show that all bout duration distributions are near power-law or
truncated power-law types. 
We also find the self-dependence of behavioural time-series by quantifying
rigorously mutual information between behaviour now and behaviour in the future,
and ask how this mutual information decays with time.
We show that the memory of a time-series of behaviour decays as
a power-law up to a point (around 1000-3000 s), after which it changes to a more
typical exponential decay. We show this in many different ways (check out our
pre-print above!)

We use the module
[`powerlaw`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777)
for distribution fitting. 
Hazard functions are estimated by our code based on the definition of a hazard
function.
Mutual information is computed using `sklearn`, and a custom implementation of
a Bialek finite-size correction is used to extrapolate its true value.
Data for this project comes from the [Communication and
Coordination Across Scales](https://www.movecall.group/) interdisciplinary
group.


# Dependencies and prerequisites

This software has been written in python 3.10 on and for a Linux operating
system. You will not need expensive supercomputers to run this code, it should
work on any personal computer (tested on i7-11th Gen, 16G RAM). However, it will
take very long. A more powerful computer will help to get these analyses
replicated in a reasonable time. At the end there are tips for using this code
on a less powerful computer. I have also tried to
make this as OS-agnostic and IDE-agnostic as possible, so you should be able to
run this on any computer directly. However, this code was written and run on
Linux systems using the Bash shell, and I have only tested it in this
configuration.

Below are details about how to install and run this software

## Pre-requisite software

The following packages have to be installed separately:

- `matplotlib`
- `numpy`
- `pandas`
- `powerlaw`
- `scikit-learn` (for metrics)
- `scipy`
- `nolds` (for DFA)

Use `pip` to install these softwares, or any other package manager of your
choice.

## Installation and setup

**NOTE**: On Linux and (possibly Mac), several below steps are automated by
running the following command.:

```
curl -sSf https://raw.githubusercontent.com/pminasandra/bout-duration-distributions/master/setup.sh | bash
```

It might fail if your version of `pip` is old; so try updating that if there is
a `pip` related error.
**If you have run the above command, skip straight to step 5.**

1. create a project directory at a location of your choice and enter it

```
mkdir /path/to/your/project
cd /path/to/your/project
```

2. Download the contents of this repository using 

`git clone https://github.com/pminasandra/bout-duration-distributions
code`

3. Also create the Data and Figures directories

```
mkdir Data
mkdir Figures
```


4. In the code/ directory, create a file called 'cwd.txt' that, on the very first
line, has the content `/path/to/your/project` 

You can do this in linux-like command lines like this:
```
echo $PWD > code/cwd.txt
```

5. After this, copy any behaviour sequence data folders into the Data/ folder.

# Usage

Run all indicated python scripts using a terminal, with the command 
`python3 <script_name>.py`

Analyses are to be done as follows:

- Running `python3 code/pkgnametbd/replicates.py` generates Markovised pseudosequences to be
  used in the remaining analyses. This step will use a lot of storage space.
  Ensure that you have enough free space on your hard-disk. **This step MUST be
  run first.**
- Running `python3 code/pkgnametbd/fitting.py` generates all bout duration
    distributions and generates tables containing AIC values. 
- Running `python3 code/pkgnametbd/survival.py` creates plots with the hazard functions for all behaviours. 
- Running `python3 code/pkgnametbd/persistence.py` performs DFA and mutual information decay analyses.
- Running `python3 code/pkgnametbd/simulate.py` runs all simulations mentioned in the paper and its appendices.

For academic colleagues, it is easy to re-work this code in your own analyses.
I am also hoping to soon make a more friendly library to incorporate these
analyses in your workflow.
Most functions also come with helpful docstrings, and the overall code structure
is modular and intuitive.
If you are familiar with basic python, the only additional thing you need to
know is about [generators](https://wiki.python.org/moin/Generators), 
a python object that is not commonly used, but
speeds up work tremendously in our case.
Useful classes are provided by `simulations/agentpool.py` and
`simulations/simulator.py`, and generally helpful functions are found in
`boutparsing.py` and `fitting.py`.

# Usage on less powerful computers

With the implementation of Markovised pseudosequences, bootstrapping, and Bialek
correction of mutual information, this code now demands a lot of computational
power. However, good replicable code should be designed so that it can be run on
any system for testing. This also keeps science equitable. Here are some tips
for running this code on a local, non-super-computer. In `config.py`

- Decrease `NUM_MARKOVISED_SEQUENCES`. Something like 3 is a good number to
  attempt replication.
- Decrease `NUM_BOOTSTRAP_REPS`. Something like 10 is a good number.
- If you desire, set `ADD_MARKOV` and `ADD_BOOTSTRAPPING` to `False`. This will
  greatly speed up your code.
- Increase of decrease `NUM_CORES` based on how many processes you can run in
  parallel on your machine.
