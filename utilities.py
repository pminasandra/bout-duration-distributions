
# Pranav Minasandra
# pminasandra.github.io
# Dec 25, 2022

import inspect
import os
import os.path

import config


def saveimg(obj, name):
    """
    Saves given object to the FIGURES directory in config.py, with file formats chosen in config.py.
    Args:
        obj: a matplotlib object with a savefig method (plt or plt.Figure)
        name (str): the name to be given to the file, *without* extensions.
    """
    dirs = [os.path.join(config.FIGURES, f) for f in config.formats]
    os.makedirs(dirs, exist_ok=True)

    for f in config.formats:
        obj.savefig(os.path.join(FIGURES, f, name+f".{f}"))


def sprint(*args, **kwargs):
    filename = str(inspect.stack()[1].filename)
    print(os.path.basename(filename)+":", *args, **kwargs)
