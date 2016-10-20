#!/bin/sh
#SBATCH --partition=%(job_type)s --qos=%(job_type)s 
#SBATCH --time=%(time)s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cpu)s
#SBATCH --mem=%(mem)s
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(output_dir)s
#SBATCH --error=%(error_dir)s
#SBATCH --mail-type=BEGIN,END,FAIL

python %(script)s
    