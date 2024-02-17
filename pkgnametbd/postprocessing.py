# Pranav Minasandra
# pminasandra.github.io
# 16 February 2024

import glob
import os.path

import pandas as pd

from pkgnametbd import config
from pkgnametbd import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

pretty_replace = {
    "insufficient_data": "Insufficient data for fitting",
    "Truncated_Power_Law": "TPL",
    "Power_Law": "PL",
    "Lognormal": "LN",
    "Exponential": "E",
    "α": "$\\alpha$",
    "λ": "$\\lambda$",
    "β": "$\\beta$",
    "μ": "$\\mu$",
    "σ": "$\\sigma$"
}

fitresults = os.path.join(config.DATA, "FitResults")
for species_ in config.species:
    sp_dir = os.path.join(fitresults, species_)

    states = list(glob.glob(os.path.join(sp_dir, "*.csv")))

    i = 0
    for state in states:
        statename = os.path.basename(state)[:-len(".csv")]
        filename = os.path.join(sp_dir, state)
        df_s = pd.read_csv(filename)
        if i == 0:
            df = pd.DataFrame({"id": df_s["id"]})
            i += 1

        df[statename] = df_s["best_fit"]

    tgtname = os.path.join(fitresults, species_ + ".tex")

    cols = list(df.columns)
    cols.remove("id")
    df = df[cols]

    content = df.to_latex(index=False)
    for string in pretty_replace:
        content = content.replace(string, pretty_replace[string])

    with open(tgtname, "w") as tgt:
        tgt.write(content)
