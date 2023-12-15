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

module load python/3.11
# pip install virtualenv

cd $HOME/image2latex/
# python -m venv i2lenv
source ./i2lenv/bin/activate
pip install -r requirements.txt

# python before_run.py 
