# IMPORTS
from huggingface_hub import login # login and notebook_login equivalent functionality
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

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

model_checkpoint = "shivaniNK8/mt5-small-finetuned-cnn-news"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("Model gotten")

dataset_file = "mail-summary.csv"
no_summary_dataset = load_dataset("csv", data_files=dataset_file)
no_summary_dataset = no_summary_dataset["train"].train_test_split(train_size=0.8, seed=22)

max_input_length = 512
max_target_length = 30

tokenized_datasets = no_summary_dataset.map(preprocess_function)
print(tokenized_datasets)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16 # default: 8
num_train_epochs = 16 # default: 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-NO3",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
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
)

trainer.train()
trainer.save_model(f"{model_name}-MODEL-1")

