import os
import mlflow
import transformers
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
from datasets import load_dataset
import numpy as np
import argparse
from azureml.core import Workspace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
from transformers import TrainingArguments
import evaluate
from box import ConfigBox

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
    # Load the SQuAD dataset...
  
if __name__ == "__main__":
    model_source_uri = os.environ.get('model_source_uri')
    test_model_name = os.environ.get('test_model_name')

    loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
    tokenizer = loaded_model.tokenizer
    batch_size = 16
    num_train_epochs = 3

    # Define TrainingArguments...
    
    dataset_name = "squad"
    batch_size = 16
    datasets, label_list, batch_size = load_custom_dataset(dataset_name, batch_size)

    tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer)

    model_name = test_model_name
    num_train_epochs = 3

    # Load the pre-trained question-answering model...
    
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

    # Save the fine-tuned model...
    
    mlflow.end_run()

    # Log the fine-tuned model...
    
    # Your additional code snippet
    question_answering = ConfigBox(
        {
        "inputs": {
          "question": "What is your Name?",
          "context": "My name is Priyanka and I am a Data Scientist."
        }
    )
    registered_model_pipeline(question_answering.inputs)
    foundation_model.name
    
    import time, sys
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        OnlineRequestSettings,
    )

    timestamp = int(time.time())
    online_endpoint_name = "qa-" + str(timestamp)
    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="Online endpoint for " + foundation_model.name + ", qa",
        auth_mode="key",
    )
    workspace_ml_client.begin_create_or_update(endpoint).wait()
    
    import time, sys
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        ProbeSettings,
    )
    # create a deployment
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=online_endpoint_name,
        model=foundation_model.id,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        liveness_probe=ProbeSettings(initial_delay=600),
    )
    workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    endpoint.traffic = {"demo": 100}
    workspace_ml_client.begin_create_or_update(endpoint).result()
    ml_client.online_endpoints
