#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V10032:1
#SBATCH --mem=12000
#SBATCH --job-name="py"
#SBATCH --output=log.out

module purge
module load Anaconda3/2022.05
module load git-lfs/2.11.0
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
python ../src/train.py -model_checkpoint north/t5_base_NCC_lm -model_type t5 -dataset_path BiasedLiar/nor_email_sum -save_online True
