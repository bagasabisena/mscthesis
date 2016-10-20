#!/bin/sh
#SBATCH --partition=short --qos=short 
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16384
#SBATCH --job-name=emptest
#SBATCH --workdir=/home/nfs/bswastanto/project
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/bswastanto/output/emptest.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/bswastanto/output/emptest.err
#SBATCH --mail-type=BEGIN,END,FAIL

python emptest.py