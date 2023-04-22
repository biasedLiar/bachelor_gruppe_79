import numpy as np
import rg
import sys
import logging
import utils
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, GPTJForCausalLM, AutoModel


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
    if x[tconf["gold_label_name"]] != None:
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
    "model_checkpoint": None,
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

# Set logging configuration
logging_level = logging.DEBUG
if tconf["logging_level"] == "info":
    logging_level = logging.INFO
elif tconf["logging_level"] == "warning":
    logging_level = logging.WARNING
elif tconf["logging_level"] == "error":
    logging_level = logging.ERROR
elif tconf["logging_level"] == "critical":
    logging_level = logging.CRITICAL
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

# Validate required configuration fields
if tconf["model_checkpoint"] is None or tconf["model_type"] is None or tconf["dataset_path"] is None:
    logging.error("Some of the required fields (model_checkpoint, model_type and dataset_path) is not set.")
    exit()

logging.info(f"Settings loaded successfully with the following settings:\n{tconf}")


# Establishing the dataset
if tconf["dataset_type"] == "hf":
    dataset = load_dataset(tconf["dataset_path"])
    logging.info(f"Dataset {tconf['dataset_path']} was loaded from HuggingFace.")
else:
    dataset = load_dataset(tconf["dataset_type"], data_files=tconf["dataset_path"])
    logging.info(f"Dataset {tconf['dataset_path']} of type {tconf['dataset_type']} was loaded from local filesystem.")
logging.debug(f"dataset = f{dataset}")


# Defining model and tokenizer. This process is model specific
if tconf["model_type"] == "t5":
    model = AutoModelForSeq2SeqLM.from_pretrained(tconf["model_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(tconf["model_checkpoint"])
    logging.info(f"Model of type T5 was loaded from checkpoint {tconf['model_checkpoint']}.")
elif tconf["model_type"] == "gptj":
    model = GPTJForCausalLM.from_pretrained(tconf["model_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(tconf["model_checkpoint"])
    logging.info(f"Model of type GPTJ was loaded from checkpoint {tconf['model_checkpoint']}.")
else:
    # using the auto classes
    model = AutoModel.from_pretrained(tconf["model_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(tconf["model_checkpoint"])
    logging.info(f"Model of type unknown was loaded from checkpoint {tconf['model_checkpoint']}.")

# Tokenizing the dataset
tokenized_datasets = dataset.map(preprocess_function)
logging.debug(f"tokenized_datasets = {tokenized_datasets}")

# Logging during training config
logging_steps = len(tokenized_datasets["train"]) // tconf["batch_size"]
logging.debug(f"logging_steps={logging_steps}")
model_name = tconf["model_checkpoint"].split("/")[-1]

# Setting up training arguments
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-log",
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
trainer.save_model(f"{model_name}-model")
logging.info(f"MODEL SAVED LOCALLY TO FILE: {model_name}-model")
