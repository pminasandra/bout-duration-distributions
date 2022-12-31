# Pranav Minasandra
# pminasandra.github.io
# Dec 31, 2022
# Nice way to end the year, with some sanity checks

import boutparsing
import fitting
import config
import utilities

def check_state_consistency():
    bdg = boutparsing.bouts_data_generator()
    states = {}
    for databundle in bdg:
        species = databundle["species"]
        print("loading data for", species, databundle["id"])
        if species not in states:
            states[species] = []
        states_, props_ = fitting.states_summary(databundle["data"])
        states[species].append(states_)

    for species in states:
        print("states check for", species)
        for i in range(len(states[species]) - 1):
            assert states[species][i] == states[species][i+1]

    print("check_state_consistency: passed.")


if __name__ == "__main__":
    check_state_consistency()
