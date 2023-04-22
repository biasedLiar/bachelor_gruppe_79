#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V10032:1
#SBATCH --job-name="py"
#SBATCH --output=log.out

module purge
module load Anaconda3/2022.05
python train.py -model_checkpoint north/t5_base_NCC_lm -model_type t5 -dataset_path BiasedLiar/nor_email_sum