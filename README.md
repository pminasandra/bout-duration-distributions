---
title: Bout duration distributions in animals
---

# Overview

This project ties together results from behavioural classifiers built using [hyenas](https://github.com/pminasandra/hyena-acc), [meerkats](https://github.com/amlan-nayak/meerkat-box), and eventually [coatis](https://github.com/pminasandra/Coati_ACC_Pipeline).
Here, I aim to find bout duration distributions for all classified behaviours for each individual of each species.
This project stems from the serendipitous discovery of heavy-tailed bout duration distributions in spotted hyenas in 2019 by Pranav Minasandra. 

Heavy-tailed distributions of bout durations could imply that self-reinforcement plays a role in behavioural dynamics at the fine scale:
such distributions have decreasing associated hazard rates, which means that the longer the animal is in a behavioural state, the less likely it becomes to exit that behaviour in the next instant.
This phenomenon is better expressed in terms of what I'm starting to call *behavioural inertia*, the tendency at a point in time for an animal to remain in its current behavioural state.
This discovery implies that wildly different mammals have rising behavioural inertias for all behavioural states.
The question is why.

This project aims to

- rigorously quantify the distributions of bout durations for all behaviours across individuals and species,
- hope that some sense comes out of this thing. 

We use the module [`powerlaw`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777) for most fitting.
Data for this project comes from the [Communication and Coordination Across Scales](https://www.movecall.group/) interdisciplinary group.


# Dependencies and prerequisites

This software has been written in python 3.8 on and for a linux operating system.
You will not need expensive supercomputers to run this code, it should work on any personal computer (tested on i7-11th Gen, 24G RAM).
I have tried to make this as OS-agnostic and IDE-agnostic as possible, so you should be able to run this on any computer directly.

## Pre-requisite software

The following packages have to be installed separately:

- `matplotlib`
- `numpy`
- `pandas`
- `powerlaw`

## Directory structure

Prepare the following tree of directories (a.k.a folders) on your computer.
Note that the data files are big, and will take up some space.

```
PROJECTROOT
├── code
├── Data
└── Figures
```

**Note**: PROJECTROOT can be changed to any name of your choice, but make sure you use the appropriate commands below.

# Usage

Download this software to your system using 
`git clone https://github.com/pminasandra/bout-duration-distributions code`
while in your PROJECTROOT directory.

Enter the `code` directory, and run `gather-data.sh` to obtain all data used in this project.

- Note 1: `gather-data.sh` is a bash script, and thus works only on linux or with git-bash maybe.
If you prefer to use an alternative system, please run the appropriate command equivalents for those in `gather-data.sh`.
- Note 2: So far, only meerkat data is available for download. 
Soon, data for all species will be put up.

Edit the file `config.py`, and change the variable "PROJECTROOT" to the path to your PROJECTROOT directory.
After this, the code will run mostly automatically.

