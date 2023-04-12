#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:A100m80:1
#SBATCH --job-name="py"
#SBATCH --output=lbm_cuda_nogptjft1.out

module purge
module load Anaconda3/2022.05
python no-gptj-ft.py
