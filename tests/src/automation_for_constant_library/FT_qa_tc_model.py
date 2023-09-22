import os
from FT_QA_model_automation import load_model
import mlflow
import transformers
import torch
import json
import pandas as pd
import datetime
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    ModelConfiguration,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer
)
from azureml.core import Workspace
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForQuestionAnswering,
    #DataCollatorForQuestionAnswering,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    AutoModelForMaskedLM,
)
from datasets import load_dataset, load_metric
import numpy as np
import evaluate

# Read configuration from JSON file
config_file = "./dataset_task.json"
with open(config_file, "r") as json_file:
    config = json.load(json_file)

dataset_name = config["dataset_name"]
batch_size = config["batch_size"]
num_train_epochs = config["num_train_epochs"]
max_length = config["max_length"]
doc_stride = config["doc_stride"]
num_labels = config["num_labels"]
task = config["task"]

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

def tokenize_and_prepare_features(dataset, tokenizer, task, max_length=max_length, doc_stride=doc_stride):
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question" if tokenizer.padding_side == "right" else "context"],
            examples["context" if tokenizer.padding_side == "right" else "question"],
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    return dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names
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
        tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer)
    else:
        tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens=True)
        metric = load_metric("seqeval")

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
        compute_metrics_fn = compute_metrics_for_ner(tokenizer, label_list)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=subset_validation_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
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
