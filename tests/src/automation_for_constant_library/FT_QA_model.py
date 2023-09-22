import os
from FT_QA_model_automation import load_model  
import mlflow
import transformers
#import os
import torch
import json
import pandas as pd
import mlflow
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
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
# import evaluate
import argparse
import os
from azureml.core import Workspace
#from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorForQuestionAnswering, Trainer
import numpy as np
from datasets import load_metric
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
import evaluate

def create_training_args(model_name, batch_size=16, num_train_epochs=3):
    return TrainingArguments(
        f"{model_name}-finetuned-qa",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
    )

def tokenize_and_align_labels(dataset, tokenizer, label_all_tokens=True):
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",  # Truncate the context, not the question
            padding="max_length",
            max_length=512,  # Adjust as needed
            return_offsets_mapping=True,
            return_tensors="pt",  # Use "pt" for PyTorch tensors
        )

        # Process the start and end positions for question answering
        labels = []
        for start_pos, end_pos in zip(examples["start_positions"], examples["end_positions"]):
            # Create start and end position labels
            label_ids = torch.zeros(tokenized_inputs["input_ids"].shape, dtype=torch.long)
            label_ids[0, start_pos] = 1  # Set 1 for start position
            label_ids[0, end_pos] = 2    # Set 2 for end position

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align(examples),
        batched=True,
    )

    return tokenized_datasets

def load_custom_dataset(dataset_name, batch_size):
    # Load the SQuAD dataset
    datasets = load_dataset(dataset_name)
    label_list = None  # No need for a label list in QA

    print(f"Loaded dataset: {dataset_name}")
    print(f"Batch Size: {batch_size}")

    return datasets, label_list, batch_size

if __name__ == "__main__":
    model_source_uri = os.environ.get('model_source_uri')
    test_model_name = os.environ.get('test_model_name')

    loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
    tokenizer = loaded_model.tokenizer

    dataset_name = "squad"
    batch_size = 16
    datasets, label_list, batch_size = load_custom_dataset(dataset_name, batch_size)

    label_all_tokens = True
    tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, label_all_tokens)

    model_name = test_model_name
    num_train_epochs = 3

    training_args = create_training_args(model_name, batch_size, num_train_epochs)
    data_collator = DataCollatorForQuestionAnswering(tokenizer)

    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

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

    # Save the fine-tuned model
    save_directory = "./FT_model_for_Question_answering"
    trainer.save_model(save_directory)
    fine_tuned_model = AutoModelForQuestionAnswering.from_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print(fine_tuned_model)
    print(tokenizer)

    mlflow.end_run()

    # Log the fine-tuned model
    model_pipeline = transformers.pipeline(task="question-answering", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    model_name = f"ft-qa-bert-base-cased"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path=model_name
        )

    registered_model = mlflow.register_model(model_info.model_uri, model_name)
    print(registered_model)
