#!/bin/bash -l 

# Set SCC project

#$ -P dl4ds
#$ -pe omp 8                                # Request 8 CPU cores
#$ -l gpus=1                                # Request 1 GPU cores
#$ -l gpu_c=6.0                             # GPU compute capability
#$ -l h_rt=1:00:00                          # Time limit
#$ -N densenet                              # job_name
#$ -j y                                     # merge stdout and stderr 
#$ -o /projectnb/dl4ds/students/tigeryi/dl4ds-spring-2025-midterm-challenge-tigeryi1998    # output folder

module load miniconda
conda activate dl4ds

python densenet.py

# job submission use: 
# qsub densenet.sh