#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="load.py"
#SBATCH --output=load.out
#SBATCH --mail-user=eliaseb@stud.ntnu.no
#SBATCH --mail-type=END


module purge
module load Anaconda3/2022.05
pip install spacy
module load git-lfs/2.11.0

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
python -m spacy download nb_core_news_sm
python load_from_hub.py
