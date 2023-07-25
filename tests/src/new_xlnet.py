from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import XLNetForSequenceClassification,XLNetTokenizer
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
from box import ConfigBox
# checkpoint = "xlnet-base-cased"
# registered_model_name = "Xlnet_registered"

# subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
# resource_group = 'sonata-test-rg'
# workspace_name = 'sonata-test-ws'
def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../config/errors.json') as f:
        return json.load(f)
        
error_messages = get_error_messages()

# model to test    
test_model_name = os.environ.get('test_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# bool to decide if we want to trigger the next model in the queue
test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue, 
# which means that the first model in the queue is triggered again after the last model is tested
test_keep_looping = os.environ.get('test_keep_looping')


def prase_logs(logs):

    # split logs by \n
    logs_list = logs.split("\n")
    # loop through each line in logs_list
    for line in logs_list:
        # loop through each error in errors
        for error in error_messages:
            # if error is found in line, print error message
            if error['parse_string'] in line:
                print (f"::error:: {error_messages['error_category']}: {line}")

def get_online_endpoint_logs(workspace_ml_client, online_endpoint_name):
    print("Deployment logs: \n\n")
    logs=workspace_ml_client.online_deployments.get_logs(name="demo", endpoint_name=online_endpoint_name, lines=100000)
    print(logs)
    prase_logs(logs)

def get_test_queue()->ConfigBox:
    #config_name = test_queue+'-test'
    #queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}"
    with open(queue_file) as f:
        content = json.load(f)
        return ConfigBox(content)

def set_tracking_uri(credential, queue):

    ws = Workspace(queue.subscription, queue.resource_group, queue.workspace)
    workspace_ml_client = MLClient(
                        credential, queue.subscription, queue.resource_group, ws
                    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    #print("Reaching here in the set tracking uri method")


def download_and_register_model(queue)->dict:
    model = XLNetForSequenceClassification.from_pretrained(queue.models)
    tokenizer = XLNetTokenizer.from_pretrained(queue.models)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="text-classification",
            artifact_path="XlNetClassification_artifact",
            registered_model_name=queue.registered_model_name
    )
    model_tokenizer = {"model":model, "tokenizer":tokenizer}
    #print("Reaching here in the download and register model methos")
    return model_tokenizer
    
