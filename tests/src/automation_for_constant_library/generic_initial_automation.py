from azureml.core import Workspace
from generic_model_download_and_register import Model
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
from azure.ai.ml import MLClient, UserIdentityConfiguration
import json
import os
from box import ConfigBox

# constants
check_override = True
WORKFLOW_PREFIX = "curated-" 

def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../../config/errors.json') as f:
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
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None


# finds the next model in the queue and sends it to github step output 
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    print ("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    index = model_list.index(test_model_name)
    #index = model_list.index(test_model_name)
    print (f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print ("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)

def create_or_get_compute_target(ml_client):
    cpu_compute_target = "STANDARD-D12"
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        print("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size="STANDARD-D12", min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()
    
    return compute


def run_azure_ml_job(code, command_to_run, environment, compute, environment_variables):
    command_job = command(
        code=code,
        command=command_to_run,
        environment=environment,
        compute=compute,
        environment_variables=environment_variables
    )
    command_job.identity = UserIdentityConfiguration()
    return command_job

def create_and_get_job_studio_url(command_job, workspace_ml_client):
   
    #ml_client = mlflow.tracking.MlflowClient()
    returned_job = workspace_ml_client.jobs.create_or_update(command_job)
    # wait for the job to complete
    workspace_ml_client.jobs.stream(returned_job.name)
    return returned_job.studio_url
# studio_url = create_and_get_job_studio_url(command_job)
# print("Studio URL for the job:", studio_url)

if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit (1)

    queue = get_test_queue()

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    print("Here is my test model name : ",test_model_name)
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue.subscription,
        resource_group_name=queue.resource_group,
        workspace_name=queue.workspace
    )
    ws = Workspace(
                subscription_id = queue.subscription,
                resource_group = queue.resource_group,
                workspace_name = queue.workspace
            )
    compute_target = create_or_get_compute_target(workspace_ml_client)
    environment_variables = {"test_model_name": test_model_name, 
           "subscription": queue.subscription,
           "resource_group": queue.resource_group,
           "workspace": queue.workspace,
           "workspcae_object": ws
           }
    command_job = run_azure_ml_job(code="./", command_to_run="python generic_model_download_and_register.py", environment="gpt2-venv:6", compute="STANDARD-D12", environment_variables=environment_variables)
    create_and_get_job_studio_url(command_job, workspace_ml_client)
    # model = Model(model_name=test_model_name, queue=queue)
    # model_and_tokenizer = model.download_and_register_model(workspace=ws)
    # print("Model config : ", model_and_tokenizer["model"].config)
