from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from azureml.mlflow import get_mlflow_tracking_uri
import mlflow
import torch
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
import json
import os

workspace = "sonata-test-ws"
subscription = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
resource_group = "sonata-test-rg"
registry = "HuggingFace"
model_name="t5-small"


def set_tracking_uri(credential):
    ws = Workspace(subscription_id = subscription, resource_group = resource_group, workspace_name = workspace)
    workspace_ml_client = MLClient(
                        credential, subscription, resource_group, ws
                    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

def download_and_register_model()->dict:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="translation_en_to_fr",
            artifact_path="t5_translation_artifact",
            registered_model_name="t5_small_for_translation"
    )
    model_tokenizer = {"model":model, "tokenizer":tokenizer}
    return model_tokenizer

if __name__ == "__main__":
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    
    except Exception as e:
        print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")
    set_tracking_uri(credential)
    model_tokenizer = download_and_register_model()
    print("Sterps completed")
    