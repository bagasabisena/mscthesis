#!/bin/sh
#SBATCH --partition=short --qos=short 
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32768
#SBATCH --job-name=emptest_lag
#SBATCH --workdir=/home/nfs/bswastanto/project
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/bswastanto/output/emptest_lag.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/bswastanto/output/emptest_lag.err
#SBATCH --mail-type=BEGIN,END,FAIL

python emptest_mlag_scale.py