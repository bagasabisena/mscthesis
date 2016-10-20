#!/bin/sh
#SBATCH --partition=short --qos=short 
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --job-name=20160313-005636
#SBATCH --output=/Users/bagas/Dropbox/msc/thesis/project/thesis/output/20160313-005636.out
#SBATCH --error=/Users/bagas/Dropbox/msc/thesis/project/thesis/output/20160313-005636.err
#SBATCH --mail-type=BEGIN,END,FAIL

python lag.py
    