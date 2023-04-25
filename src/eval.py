import numpy as np
from datasets import load_dataset
from spacy import tokenizer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GPTJForCausalLM, AutoModel
import sys
import utils
import logging
import rg

#Base config settings
tconf = {
    "model_checkpoint": None,
    "load_from_hf": True,
    "model_type": None,
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


#Load dataset
print("dataset")
print(tconf["dataset_path"])
no_summary_dataset = load_dataset(tconf["dataset_path"])

#Load model
hub_model_id = tconf["model_checkpoint"]
summarizer = pipeline("summarization", model=hub_model_id)

# Create summaries of dataset with model
texts = [row["text"] for row in no_summary_dataset["test"]]
predictions = summarizer(texts)
for i in predictions:
    i = i["summary_text"].replace("<extra_id_0>", "")
pred = [row["summary_text"] for row in predictions]




#Create standard to measure summaries against.
gold_labels = []
labels = []
for row in no_summary_dataset["test"]:
    gold_labels.append(row["goldlabel"])
    labels.append(row["label"])

#Print results.
print("Score vs human generated gold labels:")
print(rg.compareRouge(gold_labels, pred))
print()
print("Score vs gpt-generated labels:")
print(rg.compareRouge(labels, pred))



