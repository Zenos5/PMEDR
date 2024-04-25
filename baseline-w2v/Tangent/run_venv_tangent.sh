#!/bin/bash

#SBATCH --time=4:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem=80G   # 164G memory per CPU core
#SBATCH --mail-user=aw742@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=cs
#SBATCH --partition=cs

# some helpful debugging options
set -e
set -u

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.11

cd $HOME/PMEDR/baseline-w2v/Tangent/

source ./tangentenv/bin/activate

# index: index the formulas in the collection
#         flush: empty the current index
#         <directory>: directory or file containing tex and mathml documents containing formulas to index
# python indexer.py flush "../../data/MSE_dataset_full/dataset_full/math/"
python indexer.py index "../../data/MSE_dataset_full/dataset_full/math/"

# python search.py config_object query [query2, ...]
#         config_object: class name of Config object; ex: config.FMeasureConfig
#         query: query expression in latex or mathml

#         *config_object are defined in config.py and determine the host,port and score ranking
# python search.py config_object query [query2, ...]

# python mathsearch.py config_object
#         config_object: class name of Config object; ex: config.FMeasureConfig

#         The server will launch and be available on the port defined in the config object
# python mathsearch.py config_object