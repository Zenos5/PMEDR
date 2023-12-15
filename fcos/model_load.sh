#!/bin/bash

# some helpful debugging options
set -e
set -u

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.11
pip3 install virtualenv

cd $HOME/FCOS/fcos-opencv/
# cd /home/aw742/compute/
# python3 -m venv fcosenv
source ./fcosenv/bin/activate

pip3 install -r requirements.txt

python3 model_load.py 