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
module load jq zstd pigz parallel libnvidia-container enroot
CONTAINER_NAME="mycontainer"
RUN_NAME="_SGD"
OPTIM_NAME="sgd"
CHECKPOINT="checkpoint_epoch90_SGD.pth"

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/FCOS/mytag1.sqsh
fi
# run a shell
enroot start \
        --mount /lustre/scratch/usr/${USER}:/home/${USER}/compute --rw \
        --mount ${HOME}/FCOS/checkpoints:/app/checkpoints \
        --mount ${HOME}/FCOS/results:/app/results \
        mycontainer \
        ./run_training.sh --run_name=$RUN_NAME --learning_rate=0.01 --epochs=200 --optim_name=$OPTIM_NAME # the name of the command INSIDE THE CONTAINER that you want to run
