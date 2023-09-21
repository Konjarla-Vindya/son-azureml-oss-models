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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import load_dataset

def data_set(): 
    dataset = load_dataset("yelp_review_full")
    dataset["train"][5]
    print("downloaded data set-------------")
    return dataset
    

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def model():
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    model = AutoModelForSequenceClassification.from_pretrained(test_model_name, num_labels=5)
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
  ML=model()
  print("ML----------------------",ML)
  training_args = TrainingArguments(output_dir="test_trainer")
  metric = evaluate.load("accuracy")
  compute_metrics(eval_pred)
    
