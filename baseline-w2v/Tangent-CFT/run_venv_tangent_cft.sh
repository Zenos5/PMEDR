#!/bin/bash

#SBATCH --time=12:00:00   # walltime.  hours:minutes:seconds
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
# module avail python
module load python/3.11

cd $HOME/PMEDR/baseline-w2v/Tangent-CFT/

source ./tangentCFTenv/bin/activate

python -u tangent_cft_front_end.py -ds "MSE_dataset.csv" -cid 1 --wiki False -em slt_encoder.tsv --mp slt_model --rf slt_ret.tsv --qd "/TestQueries" --ri 1
# python tangent_cft_front_end.py -ds "MSE_dataset.csv" --wiki False --slt False -cid 2  -em opt_encoder.tsv --mp opt_model --tn False --rf opt_ret.tsv --qd "/TestQueries" --tn False --ri 2
# python tangent_cft_front_end.py -ds "MSE_dataset.csv" -cid 3 --wiki False -em slt_type_encoder.tsv --mp slt_type_model --rf slt_type_ret.tsv --qd "/TestQueries" --et 2 --tn False --ri 3
# python3 tangent_cft_combine_results.py