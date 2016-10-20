#!/bin/sh
#SBATCH --partition=short --qos=short 
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4096
#SBATCH --job-name=L1
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --mail-type=BEGIN,END,FAIL

python lag.py