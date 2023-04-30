from datasets import load_dataset
from transformers import pipeline
import sys
import utils
import logging
import rg

# Base config settings
tconf = {
    "model_checkpoint": None,  # must be from hf hub
    "dataset_path": None,
    "dataset_type": "hf",  # hf indicates data is loaded from huggingface
    "max_summary_length": 512,
    "input_name": "text",
    "label_name": "label",
    "gold_label_name": "goldlabel",
    "logging_level": "debug",
}

# Update config based on input
utils.update_dict_from_args(tconf, sys.argv)


# Set logging configuration
logging.basicConfig(
    level=utils.get_logging_level(tconf["logging_level"]),
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

# Validate required configuration fields
if tconf["model_checkpoint"] is None or tconf["dataset_path"] is None:
    logging.error("Some of the required fields (model_checkpoint, model_type and dataset_path) is not set.")
    exit()

# Establishing the dataset
dataset = load_dataset(tconf["dataset_path"])
logging.info(f"Dataset {tconf['dataset_path']} was loaded from HuggingFace.")

# Initialize summarizer
summarizer = pipeline("summarization", model=tconf["model_checkpoint"])

print("===================================")
print("SCORE AGAINST GOLD LABELS:")
print(rg.getRougeScore(summarizer, dataset["test"], tconf["input_name"], tconf["gold_label_name"], tconf["max_summary_length"]))
print("===================================")
print("SCORE AGAINST HUMAN LABELS:")
print(rg.getRougeScore(summarizer, dataset["test"], tconf["input_name"], tconf["label_name"], tconf["max_summary_length"]))
