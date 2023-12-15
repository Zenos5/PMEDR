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
module load python/3.11

RUN_NAME="_coco"
OPTIM_NAME="adam"
DATASET="COCO" #IBEM
# CHECKPOINT="checkpoint_epoch90_coco.pth"

cd $HOME/FCOS/fcos-opencv/
source ./fcosenv/bin/activate

python fcos_train.py \
    -n $RUN_NAME \
    -lr 0.0001 \
    -e 120 \
    -o $OPTIM_NAME \
    -d $DATASET \
    -t 0.05 \
    -s 15
# ./run_training.sh \
    # --run_name=$RUN_NAME \ 
    # --learning_rate=0.0001 \
    # --epochs=90 \
    # --optim_name=$OPTIM_NAME \
    # --dataset=$DATASET \
    # --score_thresh=0.05 \
    # --save_rate=15 \
    # --checkpoint=$CHECKPOINT
