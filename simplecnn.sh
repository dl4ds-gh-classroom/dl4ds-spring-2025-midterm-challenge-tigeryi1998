#!/bin/bash -l 

# Set SCC project

#$ -P dl4ds
#$ -pe omp 4                                # Request 4 CPU cores
#$ -l gpus=1                                # Request 1 GPU cores
#$ -l gpu_c=6.0                             # GPU compute capability
#$ -l h_rt=1:00:00                          # Time limit
#$ -N simplecnn                             # job_name
#$ -j y                                     # merge stdout and stderr 
#$ -o /projectnb/dl4ds/students/tigeryi/dl4ds-spring-2025-midterm-challenge-tigeryi1998    # output folder

module load miniconda
conda activate dl4ds

python simplecnn.py

# job submission use: 
# qsub simplecnn.sh