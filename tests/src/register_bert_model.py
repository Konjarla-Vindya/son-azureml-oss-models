from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from azureml.mlflow import get_mlflow_tracking_uri
import mlflow
import time, sys
import json
import os

checkpoint = "bert-base-uncased"
registered_model_name = "bert_registered"

def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../config/errors.json') as f:
        return json.load(f)
    
error_messages = get_error_messages()

# model to test    
test_model_name = os.environ.get('test_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# # bool to decide if we want to trigger the next model in the queue
# test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue, 
# which means that the first model in the queue is triggered again after the last model is tested
test_keep_looping = os.environ.get('test_keep_looping')

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group
def get_test_queue():
    config_name = test_queue+'-test'
    queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)
    
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model   
def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None




# we always test the latest version of the model
# def get_latest_model_version(registry_ml_client, model_name):
#     print ("In get_latest_model_version...")
#     # Getting latest model version from registry is not working, so get all versions and find latest
#     model_versions=registry_ml_client.models.list(name=model_name)
#     model_version_count=0
#     # can't just check len(model_versions) because it is a iterator
#     models = []
#     for model in model_versions:
#         model_version_count = model_version_count + 1
#         models.append(model)
    # Sort models by creation time and find the latest model
    sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
    latest_model = sorted_models[0]
    print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
    print(latest_model)
    return latest_model

# def get_instance_type(latest_model, sku_override, registry_ml_client, check_override):
#     # determine the instance_type from the sku templates available in the model properties
#     # 1. get the template name matching the sku_type
#     # 2. look up template-sku.json to find the instance_type
#     model_properties = str(latest_model.properties)
#     # escape double quotes in model_properties
#     model_properties = model_properties.replace('"', '\\"')
#     # replace single quotes with double quotes in model_properties
#     model_properties = model_properties.replace("'", '"')
#     # convert model_properties to dictionary
#     model_properties_dict=json.loads(model_properties)
#     sku_templates = model_properties_dict['skuBasedEngineIds']
#     # split sku_templates by comma into a list
#     sku_templates_list = sku_templates.split(",")
#     # find the sku_template that has sku_type as a substring
#     sku_template = next((s for s in sku_templates_list if test_sku_type in s), None)
#     if sku_template is None:
#         print (f"::error:: Could not find sku_template for {test_sku_type}")
#         exit (1)
#     print (f"sku_template: {sku_template}")
#     # split sku_template by / and get the 5th element into a variable called template_name
#     template_name = sku_template.split("/")[5]
#     print (f"template_name: {template_name}")
#     template_latest_version=get_latest_model_version(registry_ml_client, template_name)

    #print (template_latest_version.properties) 
    # split the properties by by the pattern "DefaultInstanceType": " and get 2nd element
    # then again split by " and get the first element
    # instance_type = str(template_latest_version.properties).split('"DefaultInstanceType": "')[1].split('"')[0]
    # print (f"instance_type: {instance_type}")

    # if instance_type is None:
    #     print (f"::error:: Could not find instance_type for {test_sku_type}")
    #     exit (1)

    # if check_override:
    #     if latest_model.name in sku_override:
    #         instance_type = sku_override[test_model_name]['sku']
    #         print (f"overriding instance_type: {instance_type}")
    
    # return instance_type



# def set_tracking_uri(credential):
#     subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
#     resource_group = 'sonata-test-rg'
#     workspace_name = 'sonata-test-ws'

#     ws = Workspace(subscription_id, resource_group, workspace_name)
#     workspace_ml_client = MLClient(
#                         credential, subscription_id, resource_group, ws
#                    )

    
def download_and_register_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="fill-mask",
            artifact_path="Bert_artifact",
            registered_model_name=registered_model_name
    )
    
def get_latest_model_version(registry_ml_client, test_model_name):
    model_versions = list(registry_ml_client.models.list("bert_registered"))
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



# credential = DefaultAzureCredential()
# #set_tracking_uri(credential)
# download_and_register_model()
    


def main():
    
    

    # constants
    check_override = True

    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_keep_looping is None:
        print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_keep_looping are not set")
        exit (1)

    queue = get_test_queue()

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False

    # print values of all above variables
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['subscription']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    #print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")
    
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    # connect to workspace
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue['subscription'],
        resource_group_name=queue['resource_group'],
        workspace_name=queue['workspace']
    )
mlflow.set_tracking_uri(workspace_ml_client.get_mlflow_tracking_uri())

# checkpoint = "bert-base-uncased"
# registered_model_name = "bert_registered"
    # connect to registry
    
    registry_ml_client = MLClient(
        credential=credential, 
        registry_name="sonata-test-reg"
    )

    latest_model = get_latest_model_version(registry_ml_client, test_model_name)
    #instance_type = get_instance_type(latest_model, sku_override, registry_ml_client, check_override)

    #credential = DefaultAzureCredential()
    #set_tracking_uri(credential)
    download_and_register_model()
    

if __name__ == "__main__":
    main()
