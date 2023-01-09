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
cd ${CurrDir}
exit 0
