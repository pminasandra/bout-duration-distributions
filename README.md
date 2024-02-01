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
Emily Grout,
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
patterns in behavior dynamics for all classified behaviours for each individual of
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
Furthermore, we show that the memory of a time-series of behavior decays as
a power-law up to a point (around 1000-3000 s), after which it changes to a more
typical exponential decay. We show this in many different ways (check out our
pre-print above!)

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

<!--- BROCK - REQ
Somewhere you should note which version of python and which version of each package you ran this against for the paper.
---> 

<!--- BROCK - OPT
Standard practice is to put these in requirements.txt. If your audiance is familliar with python, that's where they will look for them. If your audience is unfamilliar with python, they will need more than a list of packages to understand how to install them.
--->

The following packages have to be installed separately:

- `matplotlib`
- `numpy`
- `pandas`
- `powerlaw`
- `scikit-learn` (for metrics)
- `scipy`
- `nolds` (for DFA)


## Installation and setup
<!--- BROCK - OPT
I think these steps could be handled by the directory structure + code 
Ok, the script makes it easier, but I think you can eliminate the need for it and just ask people to clone the repo.

By "handled by the code", I mean have some module that runs at the start of each script like
"if not os.path.exists(): os.mkdir('Figures')

(The data directory the user will need to make/supply the path because they are supplying the contents I guess?)

A common pattern is to hold the path to these kind of data directories in environment variables so the user
can change them from the defaults.
---> 

**NOTE**: On Linux and (possibly Mac), several below steps are automated by running the following command:

```
curl -sSf https://raw.githubusercontent.com/pminasandra/bout-duration-distributions/master/setup.sh | bash
```

If you have run the above command, skip straight to step 5.

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

<!--- BROCK - REQ
Ok, step 5 is some serious hand waving.
What kinds of data are appropriate to use here?
What format does this data need to be in?
etc.
--->

5. After this, copy any behaviour sequence data folders into the Data/ folder.

# Usage

Run all indicated python scripts using a terminal, with the command 
`python3 <script_name>.py`

Analyses are to be done as follows:

<!--- BROCK - OPT
Suggest either making these executables or specifying to run them like "python3 fitting.py" etc.
--->
- Running `python code/pkgnametbd/fitting.py` generates all bout duration
    distributions and generates tables containing AIC values. 
- Running `python code/pkgnametbd/survival.py` creates plots with the hazard functions for all behaviours. 
- Running `python code/pkgnametbd/persistence.py` performs DFA and mutual information decay analyses.
- Running `python code/pkgnametbd/simulate.py` runs all simulations mentioned in the paper and its appendices.

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
