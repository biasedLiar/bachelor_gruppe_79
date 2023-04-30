import numpy as np
import rg
import sys
import logging
import utils
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer


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
        x[tconf["input_name"]],
        max_length=tconf["max_input_length"],
        truncation=True,
    )
    if not utils.is_empty(x[tconf["gold_label_name"]]):
        x[tconf["label_name"]] = x[tconf["gold_label_name"]]
        logging.debug("Gold label replaced label.")

    labels = tokenizer(
        x[tconf["label_name"]],
        max_length=tconf["max_target_length"],
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Default training configuration
tconf = {
    "model_checkpoint": None,  # Must be from hf
    "dataset_path": None,  # Must be from hf
    "model_save_name": None,  # Name of saved model (online). Model is also saved locally with -local suffix
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
    "save_online": "n"
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
if tconf["model_checkpoint"] is None or tconf["dataset_path"] is None or tconf["model_save_name"] is None:
    logging.error("Some of the required fields (model_checkpoint, model_type and dataset_path) is not set.")
    exit()
logging.info(f"Settings loaded successfully with the following settings:\n{tconf}")

# Establishing the dataset
dataset = load_dataset(tconf["dataset_path"])
logging.info(f"Dataset {tconf['dataset_path']} was loaded from HuggingFace.")

# Defining model and tokenizer. This process is model specific
model = AutoModelForSeq2SeqLM.from_pretrained(tconf["model_checkpoint"])
tokenizer = AutoTokenizer.from_pretrained(tconf["model_checkpoint"])
logging.info(f"Model was loaded from checkpoint {tconf['model_checkpoint']}.")

# Tokenizing the dataset
tokenized_datasets = dataset.map(preprocess_function)
logging.debug(f"tokenized_datasets = {tokenized_datasets}")

# Logging during training config
logging_steps = len(tokenized_datasets["train"]) // tconf["batch_size"]
logging.debug(f"logging_steps={logging_steps}")

# Setting up training arguments
args = Seq2SeqTrainingArguments(
    output_dir=f"{ tconf['model_save_name']}-log",
    evaluation_strategy="epoch",
    learning_rate=tconf["lr"],
    per_device_train_batch_size=tconf["batch_size"],
    per_device_eval_batch_size=tconf["batch_size"],
    weight_decay=tconf["weight_decay"],
    save_total_limit=3,
    num_train_epochs=tconf["num_train_epochs"],
    predict_with_generate=True,
    logging_steps=logging_steps,
)
logging.debug(f"args={args}")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)
logging.debug(f"tokenized_datasets = {tokenized_datasets}")

# Setting up trainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
logging.debug(f"trainer={trainer}")

logging.info("TRAINING STARTED")
trainer.train()  # Training start
logging.info("TRAINING FINISHED")
trainer.save_model(f"{tconf['model_save_name']}-local")
logging.info(f"MODEL SAVED LOCALLY TO FILE: {tconf['model_save_name']}-model")

if tconf["save_online"] == "y":
    model.push_to_hub(tconf["model_save_name"])
    tokenizer.push_to_hub(tconf["model_save_name"])
    logging.info("Model pushed to hub")
