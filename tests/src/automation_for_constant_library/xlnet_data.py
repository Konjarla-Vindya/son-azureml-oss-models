import os
from FT_initial_automation import load_model  
import mlflow
import transformers
import os
import torch
import json
import pandas as pd
import transformers
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from datasets import load_dataset
import numpy as np
# import evaluate
import argparse
import os
from azureml.core import Workspace
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import load_dataset

def data_set(): 
    dataset = load_dataset("xsum")
    dataset["train"][5]
    print("downloaded data set-------------")
    return dataset
    

def tokenize_function(examples):
    return tokenizer(examples["document"], padding="max_length", truncation=True)

def model():
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    model = AutoModelForCausalLM.from_pretrained(test_model_name, num_labels=5)
    print("model--------------------",model)
    return model
   
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
if __name__ == "__main__":
  model_source_uri=os.environ.get('model_source_uri')
  test_model_name = os.environ.get('test_model_name')
  print("test_model_name-----------------",test_model_name)
  loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
  print("loaded_model---------------------",loaded_model)
  dataset=data_set()
  tokenizer = AutoTokenizer.from_pretrained(test_model_name)
  print("tokenizer----------------------",tokenizer)
  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  print("tokenized_datasets----------",tokenized_datasets)
  small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
  small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
  model = AutoModelForCausalLM.from_pretrained(test_model_name, num_labels=5)
  training_args = TrainingArguments(output_dir="test_trainer1")
  metric = evaluate.load("rouge")
  training_args = TrainingArguments(output_dir="test_trainer1", evaluation_strategy="epoch")
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
   )
  trainer.train()
  print("training is done")
  save_directory = "./test_trainer1"
  trainer.save_model(save_directory)
  fine_tuned_model = AutoModelForCausalLM.from_pretrained(save_directory)
  tokenizer.save_pretrained(save_directory)
  fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
  model_pipeline = transformers.pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer )
timestamp_uuid = datetime.datetime.now().strftime("%m%d%H%M%f")
model_name = f"FT-TC-{test_model_name}"
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=model_pipeline,
        artifact_path=model_name
    )
registered_model = mlflow.register_model(model_info.model_uri, model_name)
print("registered_model--------------------------",registered_model)
