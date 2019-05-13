#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abeatson@princeton.edu
#SBATCH -t 48:00:00
module load cudatoolkit/8.0 cudann/cuda-8.0/5.1
. /home/abeatson/anaconda2/etc/profile.d/conda.sh
conda activate pt4
COMMAND
