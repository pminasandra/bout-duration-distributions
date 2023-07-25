# Pranav Minasandra
# pminasandra.github.io
# Dec 24, 2022

import multiprocessing as mp
import os
import os.path


#Directories
#PROJECTROOT = os.path.abspath("/media/pranav/Data1/Personal/Projects/Bout_Duration_Distributions/")
PROJECTROOT = os.path.abspath("/media/pranav/Storage1/Projects/Bout_Duration_Distributions")
if os.path.exists("cwd.txt"):
    with open("cwd.txt") as cwd:
        PROJECTROOT = cwd.read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data/")
FIGURES = os.path.join(PROJECTROOT, "Figures/")


# Species
species = ['hyena', 'meerkat', 'coati']
for s in species:
    assert s in os.listdir(DATA)
    assert os.path.isdir(os.path.join(DATA, s))


# Image saving
formats = ['png', 'svg', 'pdf']


# Distribution fitting
def all_distributions(fit):
    return {'Exponential': fit.exponential,
            'Lognormal': fit.lognormal,
            'Power_Law': fit.power_law,
            'Truncated_Power_Law': fit.truncated_power_law,
            'Stretched_Exponential': fit.stretched_exponential}

discrete = True
xmin = 2.0

distributions_to_numbers = {
    'Exponential': 0,
    'Lognormal': 1,
    'Power_Law': 2,
    'Truncated_Power_Law': 3,
    'Stretched_Exponential': 4
}


# Distribution plotting
colors = {
    'Exponential': 'cyan',
    'Lognormal': 'blue',
    'Power_Law': 'red',
    'Truncated_Power_Law': 'maroon',
    'Stretched_Exponential': 'pink'
}
fit_line_style = 'dotted'


# Survival analysis and plots
survival_plot_color = "darkblue"
survival_randomization_plot_color = "darkgreen"
survival_xscale = "log" #Use "linear" 
survival_yscale = "linear" #for typical

survival_exclude_last_few_points = True
survival_num_points_to_exclude = 100

# Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
NUM_CORES = 8

if __name__=="__main__":
    import utilities
    utilities.sprint(f"config.py speaking. current projectroot is {PROJECTROOT}")

