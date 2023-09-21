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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
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
    

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
# def get_library_to_load_model(self, task: str) -> str:
#         """ Takes the task name and load the  json file findout the library 
#         which is applicable for that task and retyrun it 

#         Args:
#             task (str): required the task name 
#         Returns:
#             str: return the library name
#         """
#         try:
#             with open(FILE_NAME) as f:
#                 model_with_library = ConfigBox(json.load(f))
#                 print(f"scoring_input file:\n\n {model_with_library}\n\n")
#         except Exception as e:
#             print(
#                 f"::warning:: Could not find scoring_file: {model_with_library}. Finishing without sample scoring: \n{e}")
#         return model_with_library.get(task)

# def download_model_and_tokenizer(self, task: str) -> dict:
#         model_library_name = self.get_library_to_load_model(task=task)
#         print("Library name is this one : ", model_library_name)
#         # Load the library from the transformer
#         model_library = getattr(transformers, model_library_name)
#         # From the library load the model
#         model_name = loaded_model.model 
#         model_name.config.num_labels = 6 
#         model_name.classifier = torch.nn.Linear(model_name.config.hidden_size, model_name.config.num_labels) 
#         Text_classification_model = model_library.from_pretrained(
#             config=model_name.config) 
#         Text_classification_model.load_state_dict(model_name.state_dict(), strict=False)
     
if __name__ == "__main__":
  model_source_uri=os.environ.get('model_source_uri')
  test_model_name = os.environ.get('test_model_name')
  print("test_model_name-----------------",test_model_name)
  loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
  print("loaded_model---------------------",loaded_model)
  data_set()
  tokenizer = AutoTokenizer.from_pretrained(test_model_name)
  print("tokenizer----------------------",tokenizer)
  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  print("tokenized_datasets----------",tokenized_datasets)
