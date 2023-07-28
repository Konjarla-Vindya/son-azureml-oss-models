# #import required libraries
# from azure.ai.ml import MLClient
# # from azure.identity import (
# #     DefaultAzureCredential,
# #     InteractiveBrowserCredential,
# #     ClientSecretCredential
# # )
# from azure.identity import DefaultAzureCredential,AzureCliCredential 
# from azureml.core import Workspace
# from azure.ai.ml.entities import AmlCompute
# from azure.ai.ml import command, Input
# import mlflow

# #Enter details of your Azure Machine Learning workspace
# subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
# resource_group = 'sonata-test-rg'
# workspace = 'sonata-test-ws'

# def connect_to_workspace():
#     #connect to the workspace
#     ws = Workspace(subscription_id, resource_group, workspace)
#     mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# def specify_compute(ml_client):
#     # specify aml compute name.
#     cpu_compute_target = "cpu-cluster"

#     try:
#         compute = ml_client.compute.get(cpu_compute_target)
#         print(compute)
#         ml_client.compute.get(cpu_compute_target)
#     except Exception:
#         print("Creating a new cpu compute target...")
#         compute = AmlCompute(
#             name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
#         )
#         ml_client.compute.begin_create_or_update(compute).result()

# def define_command():
#     # define the command
#     command_job = command(
#         code="./",
#         command="python Bert.py",
#         #--cnn_dailymail ${{inputs.cnn_dailymail}}",
#         environment="gpt2-venv:8", #"EnvTest:1",
#         # inputs={
#         #     "cnn_dailymail": Input(
#         #         type="uri_file",
#         #         path="https://datasets-server.huggingface.co/rows?dataset=cnn_dailymail&config=3.0.0&split=validation&offset=0&limit=5",
#         #     )
#         # },
#         compute="cpu-cluster",
#     )
#     return command_job

# if __name__ == "__main__":
#     try:
#         credential = AzureCliCredential()
#         credential.get_token("https://management.azure.com/.default")
#     except Exception as e:
#         print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")

#     ml_client = MLClient(
#         credential=credential,
#         subscription_id="80c77c76-74ba-4c8c-8229-4c3b2957990c",
#         resource_group_name="sonata-test-rg",
#         workspace_name="sonata-test-ws"
#         )
    
#     connect_to_workspace()
#     specify_compute(ml_client)
#     command_job = define_command()
#     # # submit the command
#     returned_job = ml_client.jobs.create_or_update(command_job)
#     # # get a URL for the status of the job
#     # returned_job.studio_url




from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
import mlflow
import os
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from azureml.mlflow import get_mlflow_tracking_uri
import mlflow
import time, sys
import json
import os


def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../../config/errors.json') as f:
        return json.load(f)
    
error_messages = get_error_messages()
test_model_name = os.environ.get('test_model_name')
test_sku_type = os.environ.get('test_sku_type')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_queue = os.environ.get('test_queue')
test_set = os.environ.get('test_set')
test_keep_looping = os.environ.get('test_keep_looping')

def get_test_queue():
    config_name = test_queue+'-test'
    queue_file1 = f"../../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)
    

def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

def set_next_trigger_model(queue):
    print ("In set_next_trigger_model...")
    index = queue['models'].index(test_model_name)
    print (f"index of {test_model_name} in queue: {index}")
    if index < len(queue['models']) - 1:
        next_model = queue['models'][index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print ("::warning:: finishing the queue")
            next_model = ""
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def get_latest_model_version(registry_ml_client, model_name):
    print ("In get_latest_model_version...")
    model_versions=registry_ml_client.models.list(name=model_name)
    model_version_count=0
    models = []
    for model in model_versions:
        model_version_count = model_version_count + 1
        models.append(model)
    sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
    latest_model = sorted_models[0]
    print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
    print(latest_model)
    return latest_model

    if check_override:
        if latest_model.name in sku_override:
            instance_type = sku_override[test_model_name]['sku']
            print (f"overriding instance_type: {instance_type}")


def create_or_get_compute_target(cpu_compute_target):
    #cpu_compute_target = "cpu-cluster"
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        print("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()
    
    return compute


def run_azure_ml_job(code, command, environment, compute):
    command_job = command(
        code=code,
        command=command,
        environment=environment,
        compute=compute,
    )
    return command_job

def create_and_get_job_studio_url(command_job):
   
    #ml_client = mlflow.tracking.MlflowClient()
    returned_job = ml_client.jobs.create_or_update(command_job)
    return returned_job.studio_url
# studio_url = create_and_get_job_studio_url(command_job)
# print("Studio URL for the job:", studio_url)


def main():
    cpu_compute_target = "cpu-cluster"
    # check_override = True

    # if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
    #     print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
    #     exit (1)

    queue = get_test_queue()

    # sku_override = get_sku_override()
    # if sku_override is None:
    #     check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)

    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['subscription']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")
    
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue['subscription'],
        resource_group_name=queue['resource_group'],
        workspace_name=queue['workspace']
    )
    ws = Workspace(subscription_id=queue['subscription'],
        resource_group=queue['resource_group'],
        workspace_name=queue['workspace'])

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    registry_ml_client = MLClient(
        credential=credential, 
        registry_name=queue['registry']
    )


    latest_model = get_latest_model_version(registry_ml_client, test_model_name)
    #download_and_register_model()
    
    compute_target = create_or_get_compute_target(cpu_compute_target)
    run_azure_ml_job(code="./", command="python Bert.py", environment="gpt2-venv:8", compute="cpu-cluster")
    create_and_get_job_studio_url(command_job)
    

if __name__ == "__main__":
    main()

