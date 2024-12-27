#!/bin/bash

#SBATCH --partition=C0
#SBATCH --job-name=MakeSegmentationPatches
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --output=outputs/slurm-segmentation_patches-%A.out

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python build_segmentation_patches.py
