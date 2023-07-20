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
# Replace with your Azure ML workspace details
# subscription_id = "bb9cf94f-f06a-49eb-a8e9-e63654d7257b"
# resource_group = "Free"
# workspace_name = "Trial"
# credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
checkpoint = "xlnet-base-cased"
registered_model_name = "Xlnet_registered"

subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
resource_group = 'sonata-test-rg'
workspace_name = 'sonata-test-ws'
test_set = os.environ.get('test_set')

def set_tracking_uri(credential):

    ws = Workspace(subscription_id, resource_group, workspace_name)
    workspace_ml_client = MLClient(
                        credential, subscription_id, resource_group, ws
                    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    #print("Reaching here in the set tracking uri method")


def download_and_register_model()->dict:
    model = XLNetForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = XLNetTokenizer.from_pretrained(checkpoint)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="text-classification",
            artifact_path="XlNetClassification_artifact",
            registered_model_name=registered_model_name
    )
    model_tokenizer = {"model":model, "tokenizer":tokenizer}
    #print("Reaching here in the download and register model methos")
    return model_tokenizer
    
def get_latest_version_model(registry_ml_client):
    model_versions = list(registry_ml_client.models.list(name=registered_model_name))
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
def test_infernce(latest_model):
    pass

def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

def get_instance_type(latest_model, sku_override, registry_ml_client, check_override):
    # determine the instance_type from the sku templates available in the model properties
    # 1. get the template name matching the sku_type
    # 2. look up template-sku.json to find the instance_type
    model_properties = str(latest_model.properties)
    # escape double quotes in model_properties
    model_properties = model_properties.replace('"', '\\"')
    # replace single quotes with double quotes in model_properties
    model_properties = model_properties.replace("'", '"')
    # convert model_properties to dictionary
    model_properties_dict=json.loads(model_properties)
    sku_templates = model_properties_dict['skuBasedEngineIds']
    # split sku_templates by comma into a list
    sku_templates_list = sku_templates.split(",")
    # find the sku_template that has sku_type as a substring
    sku_template = next((s for s in sku_templates_list if checkpoint in s), None)
    if sku_template is None:
        print (f"::error:: Could not find sku_template for {checkpoint}")
        exit (1)
    print (f"sku_template: {sku_template}")
    # split sku_template by / and get the 5th element into a variable called template_name
    template_name = sku_template.split("/")[5]
    print (f"template_name: {template_name}")
    template_latest_version=get_latest_model_version(registry_ml_client, template_name)

    #print (template_latest_version.properties) 
    # split the properties by by the pattern "DefaultInstanceType": " and get 2nd element
    # then again split by " and get the first element
    instance_type = str(template_latest_version.properties).split('"DefaultInstanceType": "')[1].split('"')[0]
    print (f"instance_type: {instance_type}")

    if instance_type is None:
        print (f"::error:: Could not find instance_type for {checkpoint}")
        exit (1)

    if check_override:
        if latest_model.name in sku_override:
            instance_type = sku_override[test_model_name]['sku']
            print (f"overriding instance_type: {instance_type}")
    
    return instance_type


def create_online_endpoint(workspace_ml_client, endpoint):
    print ("In create_online_endpoint...")
    try:
        workspace_ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    except Exception as e:
        print (f"::error:: Could not create endpoint: \n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        exit (1)

    print(workspace_ml_client.online_endpoints.get(name=endpoint.name))

def create_online_deployment(workspace_ml_client, endpoint, instance_type, latest_model):
    print ("In create_online_deployment...")
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=endpoint.name,
        model=latest_model.id,
        instance_type=instance_type,
        instance_count=1,
    )
    try:
        workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        get_online_endpoint_logs(workspace_ml_client, endpoint.name)
        workspace_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    # online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for demo deployment
    endpoint.traffic = {"demo": 100}
    try:
        workspace_ml_client.begin_create_or_update(endpoint).result()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        get_online_endpoint_logs(workspace_ml_client, endpoint.name)
        workspace_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    print(workspace_ml_client.online_deployments.get(name="demo", endpoint_name=endpoint.name))
    
if __name__ == "__main__":
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    #credential = AzureCliCredential()
    except Exception as e:
        print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")
    set_tracking_uri(credential)
    model_tokenizer = download_and_register_model()
    # connect to registry
    # registry_ml_client = MLClient(
    #     credential=credential, 
    #     registry_name="sonata-test-reg"
    # )
    registry_ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    latest_model = get_latest_version_model(registry_ml_client)
    model = model_tokenizer["model"]
    tokenizer = model_tokenizer["tokenizer"]
    inputs = tokenizer("Hello, my dog is cute", "The movie was good", return_tensors="pt")
    output = model(**inputs)
    predictions = torch.nn.functional.softmax(output.logits, dim=-1)
    print(f'Predicted class: {predictions}')
    
    timestamp = int(time.time())
    online_endpoint_name = "hf-ep-" + str(timestamp)
    print (f"online_endpoint_name: {online_endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
    sku_override = get_sku_override()
    # constants
    check_override = True
    instance_type = get_instance_type(latest_model, sku_override, registry_ml_client, check_override)
    create_online_endpoint(registry_ml_client, endpoint)
    create_online_deployment(registry_ml_client, endpoint, instance_type, latest_model)
    
