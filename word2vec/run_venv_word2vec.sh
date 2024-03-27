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

cd $HOME/Word2vec/

source ./w2venv/bin/activate

# python pytorch_word2vec/CBOW_on_negative_sampling/input_data.py
# python pytorch_word2vec/CBOW_on_negative_sampling/word2vec.py
# python pytorch_word2vec/CBOW_on_negative_sampling/test.py
python word2vec/word2vec.py