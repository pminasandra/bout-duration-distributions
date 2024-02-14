#!/bin/bash
#
# Pranav Minasandra
# pminasandra.github.io
# July 26, 2023
#
# Usage:
# curl -sSf https://raw.githubusercontent.com/pminasandra/bout-duration-distributions/master/setup.sh | bash

TOPDIR="$PWD"

mkdir bout-duration-distributions
cd bout-duration-distributions

git clone https://github.com/pminasandra/bout-duration-distributions code

mkdir Data/{,meerkat,hyena,coati}
mkdir Figures

cd code
echo $PWD > cwd.txt
python3 -m pip install setuptools wheel
python3 -m pip install -e .
python3 -m pip install -r requirements.txt

cd "$TOPDIR"
