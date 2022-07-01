#!/bin/bash
#SBATCH --mem=10G
#SBATCH -p whwong
#SBATCH -n 1
#SBATCH --time=2-00:00
#SBATCH -e error1.txt
#SBATCH -o output1.txt


ml R/4.0.2
/share/software/user/open/R/4.0.2/bin/Rscript run_causaldrf.R
