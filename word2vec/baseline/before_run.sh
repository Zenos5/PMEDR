#!/bin/bash

module load python/3.11
# pip install virtualenv

cd $HOME/PMEDR/word2vec/baseline/
# python -m venv w2vbaseenv
source ./w2vbaseenv/bin/activate
pip install -r requirements.txt

# python before_run.py 
