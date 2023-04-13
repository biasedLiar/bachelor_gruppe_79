# IMPORTS
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
    model_inputs = tokenizer(
        x["mail"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        x["summary"], 
        max_length=max_target_length, 
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# load dataset and split into train and test
dataset_file = "mail-summary.csv"
no_summary_dataset = load_dataset("csv", data_files=dataset_file)
no_summary_dataset = no_summary_dataset["train"].train_test_split(train_size=0.8, seed=22)

model_checkpoint = "north/t5_base_NCC_lm"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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
    output_dir=f"{model_name}-finetuned-NO3",
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

tokenized_datasets = tokenized_datasets.remove_columns(
    no_summary_dataset["train"].column_names
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # CHANGE TO VALIDATION WHEN READY
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{model_name}-model")
