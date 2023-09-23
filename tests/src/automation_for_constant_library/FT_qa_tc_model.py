import os
import mlflow
import transformers
import json
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    Model,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer
)
from azureml.core import Workspace
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_metric

# Read configuration from JSON file
config_file = "./dataset_task.json"
with open(config_file, "r") as json_file:
    config = json.load(json_file)

dataset_name = config["dataset_name"]
batch_size = config["batch_size"]
num_train_epochs = config["num_train_epochs"]
max_length = config["max_length"]
doc_stride = config["doc_stride"]
task = config["task"]
num_labels = config.get("num_labels", None)  # Use default value of None if not specified

def create_training_args(model_name, task, batch_size=batch_size, num_train_epochs=num_train_epochs):
    return TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
    )

def load_custom_dataset(dataset_name, task, batch_size):
    datasets = load_dataset(dataset_name)

    if task == "ner":
        label_list = datasets["train"].features[f"{task}_tags"].feature.names
    else:
        label_list = None

    print(f"Loaded dataset: {dataset_name}")
    print(f"Task: {task}")
    print(f"Label List: {label_list}")
    print(f"Batch Size: {batch_size}")

    return datasets, label_list, batch_size

def fine_tune_model(model_name, task):
    # Load model and tokenizer
    model_source_uri = os.environ.get('model_source_uri')
    loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
    tokenizer = loaded_model.tokenizer

    # Define TrainingArguments
    training_args = create_training_args(model_name, task, batch_size=batch_size, num_train_epochs=num_train_epochs)

    datasets, _, _ = load_custom_dataset(dataset_name, task, batch_size)

    if task == "qa":
        tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer, task=task)
    else:
        tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer, task=task)

    num_labels = None if task == "qa" else num_labels  # Adjust as needed
    model = (
        AutoModelForQuestionAnswering.from_pretrained(model_name)
        if task == "qa"
        else AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    )

    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

    if task == "qa":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=subset_validation_dataset,
            tokenizer=tokenizer,
        )
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=subset_validation_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    fine_tune_results = trainer.train()
    print(fine_tune_results)

    evaluation_results = trainer.evaluate()
    print(evaluation_results)

    save_directory = f"./fine_tuned_model_{task}"
    trainer.save_model(save_directory)
    fine_tuned_model = (
        AutoModelForQuestionAnswering.from_pretrained(save_directory)
        if task == "qa"
        else AutoModelForTokenClassification.from_pretrained(save_directory)
    )
    tokenizer.save_pretrained(save_directory)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print(fine_tuned_model)
    print(tokenizer)
    mlflow.end_run()

    model_pipeline = (
        transformers.pipeline(task="question-answering", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
        if task == "qa"
        else transformers.pipeline(task="token-classification", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    )
    model_name = f"ft-{task}-{model_name}"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path=model_name
        )

    registered_model = mlflow.register_model(model_info.model_uri, model_name)
    print(registered_model)

if __name__ == "__main__":
    model_name = os.environ.get('test_model_name')
    fine_tune_model(model_name, task)
