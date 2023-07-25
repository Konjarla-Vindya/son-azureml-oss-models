from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import DistilBertTokenizer, DistilBertModel
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
# checkpoint = "distilbert-base-uncased"
# registered_model_name = "distilbert_registered"

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
    model = DistilBertModel.from_pretrained(queue.models)
    tokenizer = DistilBertTokenizer.from_pretrained(queue.models)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="text-classification",
            artifact_path="XlNetClassification_artifact",
            registered_model_name=queue.registered_model_name
    )
    model_tokenizer = {"model":model, "tokenizer":tokenizer}
    #print("Reaching here in the download and register model methos")
    return model_tokenizer
    
def get_latest_version_model(registry_ml_client, queue):
    model_versions = list(registry_ml_client.models.list(name=queue.registered_model_name))
    #print(f"Here are the registered model versions : {model_versions}")
    model_version_count=0
    if len(model_versions) == 0:
        print("There is no previously registered model")
    else:
        models = []
        for model in model_versions:
            model_version_count = model_version_count + 1
            models.append(model)
        # Sort models by creation time and find the latest model
        sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
        latest_model = sorted_models[0]
        print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
        print(latest_model)
        return latest_model
    return None
def test_infernce(model_tokenizer):
    model = model_tokenizer["model"]
    tokenizer = model_tokenizer["tokenizer"]
    inputs = tokenizer("Hello, my dog is cute", "The movie was good", return_tensors="pt")
    output = model(**inputs)
    predictions = torch.nn.functional.softmax(output.logits, dim=-1)
    print(f'Predicted class: {predictions}')

def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

def create_online_endpoint(registry_ml_client, endpoint):
    print ("In create_online_endpoint...")
    try:
        registry_ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    except Exception as e:
        print (f"::error:: Could not create endpoint: \n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        exit (1)

    print(registry_ml_client.online_endpoints.get(name=endpoint.name))
def create_online_deployment(registry_ml_client, endpoint, latest_model):
    print ("In create_online_deployment...")
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=endpoint.name,
        model=latest_model.id,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    try:
        registry_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        get_online_endpoint_logs(registry_ml_client, endpoint.name)
        registry_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    # online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for demo deployment
    endpoint.traffic = {"demo": 100}
    try:
        registry_ml_client.begin_create_or_update(endpoint).result()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        get_online_endpoint_logs(registry_ml_client, endpoint.name)
        registry_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    print(registry_ml_client.online_deployments.get(name="demo", endpoint_name=endpoint.name))
    
if __name__ == "__main__":
    queue = get_test_queue()
    sku_override = get_sku_override()
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    #credential = AzureCliCredential()
    except Exception as e:
        print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")
    set_tracking_uri(credential, queue)
    model_tokenizer = download_and_register_model(queue)
    # connect to registry
    # registry_ml_client = MLClient(
    #     credential=credential, 
    #     registry_name="sonata-test-reg"
    # )
    # registry_ml_client = MLClient(
    #     credential=credential, 
    #     registry_name=queue.registry
    # )
    registry_ml_client = MLClient(
        credential= credential,
        subscription_id = queue.subscription, 
        resource_group_name = queue.resource_group, 
        workspace_name=queue.workspace
    )
    latest_model = get_latest_version_model(registry_ml_client, queue)
    test_infernce(model_tokenizer)
    
    # Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
    timestamp = int(time.time())
    online_endpoint_name = "xlnet-classification" + str(timestamp)
    print (f"online_endpoint_name: {online_endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
    create_online_endpoint(registry_ml_client, endpoint)
    create_online_deployment(registry_ml_client, endpoint, latest_model)
    
