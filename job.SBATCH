#!/bin/bash
#
#SBATCH --job-name=resnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:4

module purge

singularity exec --nv \
                 --overlay /scratch/tp2231/pytorch/overlay-50G-10M.ext3:ro \
                 /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
                 /bin/bash -c "source /ext3/env.sh; python train.py"
