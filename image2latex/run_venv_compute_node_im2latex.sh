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

DATAPATH="im2latex/im2latex-230k"
IMGPATH="im2latex/im2latex-230k/formula_images"
PREDIMGPATH="im2latex/im2latex-230k/formula_images"
# DATAPATH="im2latex/im2latex-100k"
# IMGPATH="im2latex/im2latex-100k/formula_images"
# PREDIMGPATH="im2latex/im2latex-100k/formula_images"
DATASET="230k"
# DATASET="100k"
DECODE_TYPE="beamsearch"
CHECKPOINT="lightning_logs/version_5/checkpoints/epoch=83-step=13188.ckpt"
cd $HOME/image2latex/
# python -m venv i2lenv
source ./i2lenv/bin/activate

python main.py \
    --batch-size 4 \
    --data-path $DATAPATH \
    --img-path $IMGPATH \
    --predict-img-path $PREDIMGPATH \
    --dataset $DATASET \
    --test \
    --decode-type $DECODE_TYPE \
    --max-epochs 100 \
    --ckpt-path $CHECKPOINT
