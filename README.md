--- 
title: Bout duration distributions in animals 
author: Pranav Minasandra 
---

# Details
The analyses in this code forms the basis of our [upcoming preprint](), <full title here>.

## People who worked actively on this code
[Pranav Minasandra](https://pminasandra.github.io)

## People whose contributions were necessary for this project to get going
[Ariana Strandburg-Peshkin](https://cocomo.group),
Amlan Nayak, Emily Grout, and many many others.

# Overview

This project ties together results from behavioural classifiers built using
[hyenas](https://github.com/pminasandra/hyena-acc),
[meerkats](https://github.com/amlan-nayak/meerkat-box),
[coatis](https://github.com/pminasandra/Coati_ACC_Pipeline). Here, I find
bout duration distributions for all classified behaviours for each individual of
each species. This project stems from the serendipitous discovery of
heavy-tailed bout duration distributions in spotted hyenas in 2019 by Pranav
Minasandra. 

Heavy-tailed distributions of bout durations could imply that self-reinforcement
plays a role in behavioural dynamics at the fine scale: such distributions have
decreasing associated hazard rates, which means that the longer the animal is in
a behavioural state, the less likely it becomes to exit that behaviour in the
next instant. This discovery implies that wildly
different mammals have decreasing hazard rates for all behavioural states.
We also show that all bout duration distributions are near power-law or
truncated power-law types. 

We use the module
[`powerlaw`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777)
for most fitting. 
Hazard functions are estimated by our code based on the definition of a hazard
function.
Data for this project comes from the [Communication and
Coordination Across Scales](https://www.movecall.group/) interdisciplinary
group.


# Dependencies and prerequisites

This software has been written in python 3.10 on and for a Linux operating
system. You will not need expensive supercomputers to run this code, it should
work on any personal computer (tested on i7-11th Gen, 16G RAM). I have tried to
make this as OS-agnostic and IDE-agnostic as possible, so you should be able to
run this on any computer directly.

Below are details about how to install and run this software

## Pre-requisite software

The following packages have to be installed separately:

- `matplotlib`
- `numpy`
- `pandas`
- `powerlaw`


## Installation

Download the contents of this repository using 

`git clone https://github.com/pminasandra/bout-duration-distributions
bout-duration-distributions`

Then, enter the folder just cloned and run the setup.sh script:

```
cd bout-duration-distributions/
chmod +x setup.sh
./setup.sh
```

This will set up necessary directories on your computer. Then, enter the code
directory and run the gather-data.sh script to obtain the behaviour sequences
necessary to run this software.

```
cd code/
chmod +x gather-data.sh
./gather-data.sh
```


# Usage

Run all indicated python scripts using a terminal, with the command `python3
<script_name>.py`

Upon release, there will be a `main.py`, running which will perform all the
analyses needed for our paper. Running `fitting.py` generates all bout duration
distributions and generates tables containing AIC values. Running `survival.py`
creates plots with the hazard functions for all behaviours. Running
`simulate.py` runs all simulations mentioned in the paper and its appendices.

For academic colleagues, it is easy to re-work this code in your own analyses.
Most functions also come with helpful docstrings, and the overall code structure
is modular and intuitive.
If you are familiar with basic python, the only additional thing you need to
know is about [generators](https://wiki.python.org/moin/Generators), 
a python object that is not commonly used, but
speeds up work tremendously in our case.
Useful classes are provided by `simulations/agentpool.py` and
`simulations/simulator.py`, and generally helpful functions are found in
`boutparsing.py` and `fitting.py`.

# Other uses

This software also computes [behavioural inertia](docs/behavioural-inertia.md),
and performs simulations to [demonstrate](docs/simulations.md) that classifiers
don't cause heavy-tailed distributions in their predictions.
