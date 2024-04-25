#!/bin/bash

module load python/3.11
# pip install virtualenv

cd $HOME/PMEDR/baseline-w2v/Tangent-CFT/
# python -m venv tangentCFTenv
source ./tangentCFTenv/bin/activate
pip install -r requirements.txt

