import numpy as np
from datasets import load_dataset
from spacy import tokenizer
from transformers import pipeline

import rg

hub_model_id = "BiasedLiar/mt5-sum-v1-finetuned-NO3"
summarizer = pipeline("summarization", model=hub_model_id)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rg.compareRouge(decoded_labels, decoded_preds)
    return result

no_summary_dataset = load_dataset("BiasedLiar/nor_email_sum")

def print_summary(idx):
    text = no_summary_dataset["test"][idx]["text"]
    realSum = no_summary_dataset["test"][idx]["label"]
    realGoldSum = no_summary_dataset["test"][idx]["goldlabel"]
    if realGoldSum != None and realGoldSum.strip() != "":
        realSum = realGoldSum

    summary = summarizer(text)
    print(text)
    print()
    print(realSum)
    print()
    print(summary)

print_summary(50)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
texts = [row["text"] for row in no_summary_dataset["test"]]
predictions = summarizer(texts)
print(predictions)


for i in predictions:
    i = i["summary_text"].replace("<extra_id_0>", "")

pred = [row["summary_text"] for row in predictions]
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
labels = []

for row in no_summary_dataset["test"]:
    realSum = row["label"]
    realGoldSum = row["goldlabel"]
    if realGoldSum != None and realGoldSum.strip() != "":
        print(realSum)
        print(realGoldSum)
        print()
        realSum = realGoldSum
    labels.append(realGoldSum)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

print(len(labels))
print(len(pred))
for i in range(len(labels)):
    print("real: " + labels[i])
    print("pred: " + pred[i])
    score = rg.compareRouge(labels[i], pred[i])
    print("Score: ", str(score))
    print()
print(rg.compareRouge(labels, pred))

