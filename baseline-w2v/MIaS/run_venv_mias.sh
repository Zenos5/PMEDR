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
module load python/3.11 jdk maven/3.6

cd $HOME/PMEDR/baseline-w2v/MIaS/

# printf "\n<===MathMLCan===>\n"
# cd MathMLCan/
# mvn clean install
# cd ..

# printf "\n<===MathMLUnificator===>\n"
# cd MathMLUnificator/
# mvn clean install
# cd ..

# printf "\n<===MIaSMath===>\n"
# cd MIaSMath/
# mvn clean install
# cd ..

printf "\n<===MIaS===>\n"
cd MIaS/
mvn clean install
cd ..

# printf "\n<===WebMIaS===>\n"
# cd WebMIaS/
# mvn clean install
# cd ..
