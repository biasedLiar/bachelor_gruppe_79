#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="eval.py"
#SBATCH --output=eval.out
#SBATCH --mail-user=eliaseb@stud.ntnu.no
#SBATCH --mail-type=END


module purge
module load Anaconda3/2022.05
pip install spacy

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
python -m spacy download nb_core_news_sm
python eval.py -model_checkpoint BiasedLiar/mt5-small-finetuned-cnn-news-finetuned-NO3 -dataset_path BiasedLiar/nor_email_sum