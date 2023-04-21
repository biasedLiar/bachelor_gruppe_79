#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=20:05:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120000
#SBATCH --job-name="mrm8488/mt5-base-finetuned-notes-summaries"
#SBATCH --output=mrm8488/mt5-base-finetuned-notes-summaries.out
#SBATCH --mail-user=eliaseb@stud.ntnu.no
#SBATCH --mail-type=END


module purge
module load Anaconda3/2022.05
pip install spacy
module load git-lfs/2.11.0

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
python -m spacy download nb_core_news_sm
python no-mt5-huggingface.py
