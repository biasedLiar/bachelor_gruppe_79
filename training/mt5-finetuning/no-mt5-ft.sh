#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:V10032:1
#SBATCH --job-name="mt5-ft.py"
#SBATCH --output=log.out
#SBATCH --mail-user=eliaseb@stud.ntnu.no
#SBATCH --mail-type=ALL


module purge
module load Anaconda3/2022.05
pip install spacy
python -m spacy download nb_core_news_sm
python no-mt5-ft.py
