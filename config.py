# Pranav Minasandra
# pminasandra.github.io
# Dec 24, 2022

import os
import os.path


#Directories
PROJECTROOT = os.path.abspath("..")
DATA = os.path.join(PROJECTROOT, "Data/")
FIGURES = os.path.join(PROJECTROOT, "Figures/")


# Species
species = ['hyena', 'meerkat'] #TODO: Eventually, 'coati' also goes here
for s in species:
    assert s in os.listdir(DATA)
    assert os.path.isdir(os.path.join(DATA, s))


# Image saving
formats = ['png', 'svg', 'pdf']

if __name__=="__main__":
    import utilities
    utilities.sprint(f"config.py speaking. current projectroot is {PROJECTROOT}")


# Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
