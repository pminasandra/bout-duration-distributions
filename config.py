# Pranav Minasandra
# pminasandra.github.io
# Dec 24, 2022

import os
import os.path


#Directories
PROJECTROOT = os.path.abspath("/media/pranav/Data1/Personal/Projects/Bout_Duration_Distributions/")
DATA = os.path.join(PROJECTROOT, "Data/")
FIGURES = os.path.join(PROJECTROOT, "Figures/")


# Species
#species = ['hyena', 'meerkat', 'coati', 'blackbuck']
species = ['blackbuck']
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


# Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False


# Blackbuck specific configuration
Blackbuck_Disruptive_Events = ["D1", "D5", "D6", "D7", "D10", "D13", "MM", "MF", "EX", "EXoL", "Z", "DEP", "DN"] # TODO Confirm what Z, DEP is
Blackbuck_Contiguous_Events = ["DR", "D2", "D3", "D4", "D14", "D12"]
Blackbuck_States = ["F", "M", "S", "L", "oL", "D8", "D9", "D11"]

Blackbuck_Reduced_State = {
    "F": "Active",
    "M": "Active",
    "S": "Inactive",
    "L": "Inactive",
    "oL": "Inactive",
    "D8": "Active",
    "D9": "Active",
    "D11": "Inactive"
}

if __name__=="__main__":
    import utilities
    utilities.sprint(f"config.py speaking. current projectroot is {PROJECTROOT}")

