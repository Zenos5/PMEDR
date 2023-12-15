#!/bin/bash

#SBATCH --time=9:00:00   # walltime.  hours:minutes:seconds
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
module load jq zstd pigz parallel libnvidia-container enroot
CONTAINER_NAME="mycontainer"
RUN_NAME="_adam"
OPTIM_NAME="adam"
CHECKPOINT=None

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/FCOS/mytag.sqsh
fi
# run a shell
enroot start \
        --mount /lustre/scratch/usr/${USER}:/home/${USER}/compute --rw \
        --mount ${HOME}/FCOS/checkpoints:/app/checkpoints \
        --mount ${HOME}/FCOS/results:/app/results \
        mycontainer \
        ./run_training.sh --run_name=$RUN_NAME --learning_rate=1e-4 --epochs=100 --optim_name=$OPTIM_NAME --score_thresh=0.05 # the name of the command INSIDE THE CONTAINER that you want to run
