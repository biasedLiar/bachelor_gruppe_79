import numpy as np
from datasets import load_dataset
from spacy import tokenizer
from transformers import pipeline
import sys
import utils

import rg

tconf = {
    "model_checkpoint": None,
    "dataset_path": None,
    "dataset_type": "hf",  # hf indicates data is loaded from huggingface
    "max_input_length": 512,
    "max_target_length": 256,
    "batch_size": 8,
    "num_train_epochs": 8,
    "lr": 5.6e-5,
    "weight_decay": 0.01,
    "input_name": "text",
    "label_name": "label",
    "gold_label_name": "goldlabel",
    "logging_level": "debug",
}
# Update config based on input
args = sys.argv
for i in range(len(args)):
    if args[i][0] == '-' and args[i][1] != '-':
        val = args[i + 1]
        if utils.is_int(val):
            val = int(val)
        elif utils.is_float(val):
            val = float(val)
        tconf[args[i][1:]] = val



print("dataset")
print(tconf["dataset_path"])

hub_model_id = tconf["model_checkpoint"]
summarizer = pipeline("summarization", model=hub_model_id)
no_summary_dataset = load_dataset(tconf["dataset_path"])

texts = [row["text"] for row in no_summary_dataset["test"]]
predictions = summarizer(texts)
for i in predictions:
    i = i["summary_text"].replace("<extra_id_0>", "")
pred = [row["summary_text"] for row in predictions]


labels = []
for row in no_summary_dataset["test"]:
    labels.append(row["goldlabel"])

print(rg.compareRouge(labels, pred))


