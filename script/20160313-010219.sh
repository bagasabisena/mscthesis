#!/bin/sh
#SBATCH --partition=short --qos=short 
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4096
#SBATCH --job-name=l1
#SBATCH --output=/Users/bagas/Dropbox/msc/thesis/project/thesis/output/l1.out
#SBATCH --error=/Users/bagas/Dropbox/msc/thesis/project/thesis/output/l1.err
#SBATCH --mail-type=BEGIN,END,FAIL

python lag.py
    