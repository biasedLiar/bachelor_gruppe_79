# IMPORTS
from huggingface_hub import login # login and notebook_login equivalent functionality
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import rg
import numpy as np




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

def preprocess_function(x):
    if x["goldlabel"] != None and x["goldlabel"].strip() != "":
        x["label"] = x["goldlabel"].strip()
    model_inputs = tokenizer(
        x["text"],
        max_length=max_input_length,
        truncation=True,
    )

    #if row != "subject" and row["goldlabel"] != None: row["label"] = row["goldlabel"]
    labels = tokenizer(
        x["label"],
        max_length=max_target_length,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# load dataset and split into train and test

#model_checkpoint = "shivaniNK8/mt5-small-finetuned-cnn-news"
#model_checkpoint = "chenhg8680/mt5-sum-v1"
model_checkpoint = "mrm8488/mt5-base-finetuned-notes-summaries"
#model_checkpoint = "nestoralvaro/mt5-base-finetuned-xsum-mlsum___summary_text_google_mt5_base"
#model_checkpoint = "nestoralvaro/mT5_multilingual_XLSum-finetuned-xsum-mlsum___summary_text"
#model_checkpoint = "SGaleshchuk/mT5-sum-news-ua"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("Model gotten")

dataset_file = "mail-summary.csv"
no_summary_dataset = load_dataset("BiasedLiar/nor_email_sum")
#no_summary_dataset = no_summary_dataset["train"].train_test_split(train_size=0.8, seed=22)

max_input_length = 512
max_target_length = 30

tokenized_datasets = no_summary_dataset.map(preprocess_function)
print(tokenized_datasets)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 8 # default: 8
num_train_epochs = 8 # default: 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"mt5-versions/{model_name}-finetuned-NO3",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(tokenized_datasets)
print(no_summary_dataset)
tokenized_datasets = tokenized_datasets.remove_columns(
    no_summary_dataset["train"].column_names
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    eval_dataset=tokenized_datasets["validation"], # CHANGE TO VALIDATION WHEN READY
    compute_metrics=compute_metrics,
)

trainer.train()
print("Computing real final score-----------------------------------------------------------")

trainer.push_to_hub()
print("------------------ferdigish-----------------------")
trainer.save_model(f"mt5-versions/{model_name}-MODEL-1")
print(model_checkpoint)

predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)


print()

