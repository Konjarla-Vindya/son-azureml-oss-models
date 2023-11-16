#from model_inference_and_deployment import ModelInferenceAndDeployemnt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from azure.ai.ml import command
import mlflow
import json
import os
import sys
from box import ConfigBox
from mlflow.tracking.client import MlflowClient
from azureml.core import Workspace, Environment
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
from azure.ai.ml.entities import AmlCompute
import time
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
import ast


check_override = True


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

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model


def get_sku_override():
    try:
        with open('../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None


# finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    print("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    index = model_list.index(test_model_name)
    #index = model_list.index(test_model_name)
    print(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def run_azure_ml_job(code, command_to_run, environment, compute, environment_variables):
    command_job = command(
        code=code,
        command=command_to_run,
        environment=environment,
        compute=compute,
        environment_variables=environment_variables
    )
    return command_job


def create_and_get_job_studio_url(command_job, workspace_ml_client):

    #ml_client = mlflow.tracking.MlflowClient()
    returned_job = workspace_ml_client.jobs.create_or_update(command_job)
    # wait for the job to complete
    workspace_ml_client.jobs.stream(returned_job.name)
    return returned_job.studio_url


def get_latest_model_version(workspace_ml_client, test_model_name):
    print("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(test_model_name))
    
    if len(version_list) == 0:
        print("Model not found in registry")
        foundation_model_name = None  # Set to None if the model is not found
        foundation_model_id = None  # Set id to None as well
    else:
        model_version = version_list[0].version
        foundation_model = workspace_ml_client.models.get(
            test_model_name, model_version)
        print(
            "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                foundation_model.name, foundation_model.version, foundation_model.id
            )
        )
        foundation_model_name = foundation_model.name  # Assign the value to a new variable
        foundation_model_id = foundation_model.id  # Assign the id to a new variable
    
    # Check if foundation_model_name and foundation_model_id are None or have values
    if foundation_model_name and foundation_model_id:
        print(f"Latest model {foundation_model_name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        print("foundation_model.name:", foundation_model_name)
        print("foundation_model.id:", foundation_model_id)
    else:
        print("No model found in the registry.")
    
    #print(f"Model Config : {latest_model.config}")
    return foundation_model


if __name__ == "__main__":
      # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    # sku_override = get_sku_override()
    # if sku_override is None:
    #     check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    # print values of all above variables
    print("Running for QA")
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")
    print("Here is my test model name : ", test_model_name)
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    print("workspace_name : ", queue.workspace)
    workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    foundation_model = get_latest_model_version(workspace_ml_client, test_model_name.lower())
  
    Finetune_tasks=foundation_model.properties.get("finetune-recommended-sku")
    print("Finetune tasks from model card:", {Finetune_tasks})
  
  
