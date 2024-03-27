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

MATH_DATADIR="../data/MSE_dataset_full/dataset_full/math/"
F2V_CHECKPOINT="doc2vec/checkpoints/f2v_CBOW/f2v_50.model"
TEXT_DATADIR="../data/MSE_dataset_full/dataset_full/text/"
D2V_CHECKPOINT="doc2vec/checkpoints/d2v_CBOW/d2v_50.model"
#If include arg [--test 1], then is set to true

#--compare 1 \
#--test 1 \
#--assess 1 \

python doc2vec/doc2vec_form2vec.py \
    --text-data-dir $TEXT_DATADIR \
    --math-data-dir $MATH_DATADIR \
    --d2v-checkpoint $D2V_CHECKPOINT \
    --f2v-checkpoint $F2V_CHECKPOINT \
    --eval 1 \
    --compare 1