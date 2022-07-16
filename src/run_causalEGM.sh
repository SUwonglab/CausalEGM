#!/bin/bash
#SBATCH --mem=10G
#SBATCH -p whwong
#SBATCH -n 1
#SBATCH -G 1
#SBATCH --time=2-00:00
#SBATCH -e error2.txt
#SBATCH -o output2.txt

ml load cuda/11.2.0
ml load cudnn/8.1.1.33
ml python/3.9.0

#time /share/software/user/open/python/3.9.0/bin/python3 main.py -c configs/sim_linear.yaml
#time /share/software/user/open/python/3.9.0/bin/python3 main.py -c configs/sim_quadratic.yaml
#time /share/software/user/open/python/3.9.0/bin/python3 main.py -c configs/sim_Hirano_Imbens.yaml

time /share/software/user/open/python/3.9.0/bin/python3 main.py -c configs/semi_acic.yaml
