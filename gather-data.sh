#!/bin/bash
# Pranav Minasandra
# Dec 25, 2022

PROJECTROOT="/media/pranav/Data1/Personal/Projects/Bout_Duration_Distributions"
CurrDir=$PWD

# Gathering hyena data
echo "Gathering hyena data"
cd "${PROJECTROOT}/Data"
mkdir -v hyena
cp -v /media/pranav/Data1/Personal/Projects/Strandburg-Peshkin\ 2019-20/Data/ClassificationsInTotal/*.csv ./hyena/ #Replace this with a git clone eventually
cd hyena
for f in $(ls); do
    sed s/time/datetime/g ${f} > tmp
    mv tmp ${f}
done

# Gathering meerkat data
echo "Gathering meerkat data"
cd "${PROJECTROOT}/Data"
if [ -d meerkat ]; then
    cd meerkat
    git pull
else
    git clone git@github.com:amlan-nayak/Meerkat_Behavior_Data.git meerkat
fi

# Gathering coati data
echo "Gathering coati data"
cd "${PROJECTROOT}/Data"
mkdir -v coati
cd coati
cp -v /media/pranav/Data1/Personal/Projects/Coati_ACC_Pipeline_2022/Data/VeDBA_States/*.csv .

# Fixing blackbuck data
echo "Data fixes for blackbuck"
if [ ! -d "${PROJECTROOT}/Data/blackbuck" ]; then
    echo "\033[1;31mERR:\033[0m The Data/blackbuck directory was not found" > /dev/stderr
    exit -1
fi
cd "${PROJECTROOT}/Data/blackbuck"
for f in $(ls *.csv); do
    echo ${f}
    sed s/TIME/datetime/g ${f} > tmp
    cp tmp ${f}
    sed s/ACT/state/g ${f} > tmp
    cp -v tmp ${f}
done

# Exit
cd ${CurrDir}
exit 0
