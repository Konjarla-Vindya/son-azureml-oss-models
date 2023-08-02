print("py started")
import os 
# Set the Hugging Face token as an environment variable
#os.environ["HF_TOKEN"] = os.environ.get('test_model_name')

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import mlflow
import pandas as pd
import numpy as np
from datasets import load_dataset
import evaluate
from azureml.core import Workspace
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments,Trainer,DistilBertForMaskedLM
import pickle
#from azure.ai.ml import MLClient
#from azureml.core import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
import mlflow
from tensorflow.keras import Model
#from azure.ai.ml.entities import AmlCompute
import time
import json
import mlflow.pytorch
import torch
tensor_1 = torch.rand (4,4)
# import huggingface_hub
# import huggingface_hub.login
print("imported")

test_model_name = os.environ.get('test_model_name')
subscription = os.environ.get('subscription')
resource_group = os.environ.get('resource_group')
workspace_name = os.environ.get('workspace')

# import os
# import json

# def create_mlflow_experiment(model_name):
#     mlflow.create_experiment(model_name)

# config_file_path = "test-distl-bert_model.json"  #"config.json"

# def read_config_file(file_path):
#     with open(file_path, 'r') as f:
#         config = json.load(f)
#     return config

# config = read_config_file(config_file_path)

# if "models" in config and isinstance(config["models"], list):
#     for model_name in config["models"]:
#         os.environ["MLFLOW_EXPERIMENT_NAME"] = model_name
#         create_mlflow_experiment(model_name)
#         print(f"Created MLflow experiment for model: {model_name}")
# else:
#     print("Error: 'models' key not found in the config file or not a list.")

os.environ["MLFLOW_EXPERIMENT_NAME"] = "DistlBertBaseUncased"
print("error fixed")

#MLFLOW_TRACKING_URI
#mlflow.set_registry_uri(EXPERIMENT_NAME)
#with mlflow.start_run(experiment_id=EXPERIMENT_NAME):

print("experiment created")
import mlflow.pytorch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

print(test_model_name)
model = AutoModelForSequenceClassification.from_pretrained(test_model_name)
tokenizer = AutoTokenizer.from_pretrained(test_model_name)
raw_ds= load_dataset("glue", "mrpc")
raw_ds['train'] = raw_ds['train'].shuffle().select(range(100))
raw_ds['test']= raw_ds['test'].shuffle().select(range(20))
raw_ds['validation']= raw_ds['validation'].shuffle().select(range(20))
metric = evaluate.load("glue", "mrpc")
dataset = raw_ds.map(
    lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True),
    batched=True,
)
dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
dataset = dataset.rename_column("label", "labels")
dataset = dataset.with_format("torch")

print("checkpoint done")

trainer_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

print("args done")

# test_dataset = load_dataset("glue","mrpc", split="test").shuffle().select(range(1000))
# train_dataset = load_dataset("glue", "mrpc", split="train").shuffle().select(range(1000))

def compute_metrics(eval_preds: EvalPrediction):
    x, y = eval_preds
    preds = np.argmax(x, -1)
    return metric.compute(predictions=preds, references=y)

print("metrics done")


    
mlflow.transformers.save_model(
    transformers_model={"model": model, "tokenizer": tokenizer },
    path="./savedmodel",
    task="fill-mask",
    registered_model_name=test_model_name
    # signature=signature,
    # input_example=data,
)
print("saved model")
#registered_model = mlflow.register_model(model_info.model_uri, model_name)

with mlflow.start_run():
   
    model_local_path = os.path.abspath("./savedmodel")
    mlflow.register_model(f"file://{model_local_path}", test_model_name)

print("registered saved model")
	
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=trainer_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
    # test_dataset=dataset["test"]
)

result = trainer.train()

print("trainer done")

#retrieve trained model

trained_model = trainer.model
trained_model.config

mlflow.end_run()	

print("Ended")

model_tokenizer = {"model":model, "tokenizer":tokenizer}

model = model_tokenizer["model"]
# Move the model to the CPU model = model.to('cpu')

# inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

# model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").to('cuda:0') 
# Prepare the input text with masked token and move it to GPU 
input_text = "I love using [MASK]'s transformers library!" 
# inputs = tokenizer(input_text, return_tensors="pt").to('cuda:0')
inputs = tokenizer(input_text, return_tensors="pt")
output=model(**inputs)
print(output)
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(f'Predicted class: {predictions}')
logits = output.logits
predicted_token_ids = torch.argmax(logits, dim=-1)
predicted_words = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
predicted_words
