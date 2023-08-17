#!/bin/bash
#
# Pranav Minasandra
# pminasandra.github.io
# July 26, 2023
#
# USAGE: Run this script right after cloning to set up appropriate directories
# as needed to produce the results in our paper.

cpwd=$PWD

cd dirname $0
mkdir code/ 
mv ./* code/
mkdir Data/{,coati,hyena,meerkat} Figures/

cd $cpwd

exit 0 
