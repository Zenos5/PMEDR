#!/bin/bash

module load python/3.11
# pip install virtualenv

cd $HOME/PMEDR/baseline-w2v/Tangent/
# python -m venv tangentenv
source ./tangentenv/bin/activate
pip install -r requirements.txt

# python before_run.py 
