#!/bin/bash

#SBATCH --time=1:00:00   # walltime.  hours:minutes:seconds
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

cd $HOME/PMEDR/word2vec/

source ./w2venv/bin/activate

DATADIR="../data/MSE_dataset_full/dataset_full/text/"
CHECKPOINT="doc2vec/checkpoints/d2v_50.model"
#If include arg [--train 1], then is set to true

python doc2vec/doc2vec.py \
    --start-epoch 0 \
    --max-epochs 50 \
    --data-dir $DATADIR \
    --checkpoint $CHECKPOINT \
    --vector-size 50 \
    --min-count 2 \
    --strategy 0 \
    --train 1 \
    --test 1 \
    --eval 1 \
    --neg-sample 5 \
    --hier-sm 0