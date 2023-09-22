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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
#from transformers.data.data_collator import DataCollatorForQuestionAnswering  # Corrected import

import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
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

def tokenize_and_prepare_features(dataset, tokenizer, max_length=384, doc_stride=128):
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
    batch_size = 16
    num_train_epochs = 3

     # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
    )

    dataset_name = "squad"
    batch_size = 16
    datasets, label_list, batch_size = load_custom_dataset(dataset_name, batch_size)

    tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer)

    model_name = test_model_name
    num_train_epochs = 3

    # Load the pre-trained question-answering model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    training_args = create_training_args(model_name, batch_size, num_train_epochs)
    #data_collator = DataCollatorForQuestionAnswering(tokenizer)

    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=subset_train_dataset,
        eval_dataset=subset_validation_dataset,
        #data_collator=data_collator,
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
