#!/bin/bash

module load python/3.11
# pip install virtualenv

cd $HOME/PMEDR/word2vec/
# python -m venv w2venv
source ./w2venv/bin/activate
pip install -r requirements.txt

# python before_run.py 
