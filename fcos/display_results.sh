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


set -e
set -u

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# pip3 install virtualenv
module load python/3.11

CHECKPOINT="checkpoints/ibem_mod/"
RESULTS="ibem_v2"

cd $HOME/PMEDR/fcos/
source ./fcosenv/bin/activate
# python3 -m venv myenv

python display_results.py \
  -p $CHECKPOINT \
  -r $RESULTS