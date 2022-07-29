#!/bin/bash
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH --mem=30G # 4 GBs RAM 
#SBATCH –p gpu-common
python trainUnet.py -train ardmayle kilbixy -test knockainey -prop 0.8 -s 512 -lr 0.01 -m 0.9 -e 10 -b 10 -f -o knockainey

