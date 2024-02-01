#!/bin/bash
#
# Pranav Minasandra
# pminasandra.github.io
# July 26, 2023
#
# Usage:
# curl -sSf https://raw.githubusercontent.com/pminasandra/bout-duration-distributions/master/setup.sh | bash

mkdir bout-duration-distributions
cd bout-duration-distributions

git clone https://github.com/pminasandra/bout-duration-distributions code

mkdir Data
mkdir Figures

echo $PWD > cwd.txt
