#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V10032:1
#SBATCH --job-name="py"
#SBATCH --output=lbm_cuda_not5ft3.out

module purge
module load Anaconda3/2022.05
python no-t5-ft.py
