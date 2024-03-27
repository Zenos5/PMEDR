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

# DATAPATH="../data/im2latex/im2latex-230k"
# IMGPATH="../data/im2latex/im2latex-230k/formula_images"
# PREDIMGPATH="../data/im2latex/im2latex-230k/formula_images"
DATAPATH="../data/im2latex/im2latex-100k"
IMGPATH="../data/im2latex/im2latex-100k/formula_images"
PREDIMGPATH="../data/im2latex/im2latex-100k/formula_images"
# DATASET="230k"
DATASET="100k"
DECODE_TYPE="beamsearch" #"greedy"
BAYESIAN=true #false if removed from arguments # --bayesian $BAYESIAN \
ENC_TYPE="conv_row_encoder" #"conv_row_encoder""resnet_row_encoder"
CHECKPOINT="lightning_logs/version_62745862/checkpoints/epoch=24-step=3925.ckpt"
# version_62758420 -> orig train full
cd $HOME/PMEDR/image2latex/
# python -m venv i2lenv
source ./i2lenv/bin/activate

python main.py \
    --batch-size 4 \
    --data-path $DATAPATH \
    --img-path $IMGPATH \
    --predict-img-path $PREDIMGPATH \
    --dataset $DATASET \
    --predict \
    --decode-type $DECODE_TYPE \
    --enc-type $ENC_TYPE \
    --max-epochs 50 \
    --ckpt-path $CHECKPOINT
