from transformers import AutoTokenizer
from azureml.core import Workspace, Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import Input, MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
# from utils.logging import get_logger
from huggingface_hub import login
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azure.ai.ml.constants import AssetTypes
from mlflow.tracking.client import MlflowClient
import time
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    Model,
)
from azureml.core.datastore import Datastore
from azureml.core import Workspace
from mlflow.tracking.client import MlflowClient
import re
from datetime import datetime
import base64
import logging

# constants
check_override = True

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



def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))


def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None

# Function to dynamically replace masking tokens
def process_input_for_fill_mask_task(file_path, mask_token):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Detect and replace the masking token based on the model's mask token
        file_content = re.sub(r'\[MASK\]', mask_token, file_content)
        #file_content = re.sub(r'<mask>', mask_token, file_content)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(file_content)

    except Exception as e:
        print(f"Error processing {file_path} for 'fill-mask' task: {str(e)}")

def get_task_specified_input(task, test_model_name):
    print("pulling inputs")
    folder_path = f"../../config/sample_inputs/{queue.registry}/{task}/batch_inputs"

    # List all file names in the folder
    file_names = os.listdir(folder_path)
    
    # Create a list to store individual input objects for each file
    inputs = []
    
    # Process each file in the folder
    for file_name in file_names:
        print("File Name:", file_name)
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Create an Input object for the file and add it to the list of inputs
            file_input = Input(path=file_path, type=AssetTypes.URI_FILE)
            # Handle the "fill-mask" task by replacing [MASK] with <mask> in the input data
            if task.lower() == "fill-mask":
                tokenizer = AutoTokenizer.from_pretrained(test_model_name)
                #tokenizer = AutoTokenizer.from_pretrained(test_model_name, trust_remote_code=True, use_auth_token=True)
                mask_token = tokenizer.mask_token  
                process_input_for_fill_mask_task(file_path, mask_token)
            # if task.lower() == "fill-mask":
            #     try:
            #         with open(file_path, 'r') as file:
            #             file_content = file.read()
                    
            #         # Replace [MASK] with <mask> in the input data
            #         file_content = file_content.replace('[MASK]', '<mask>')
                    
            #         # Write the modified content back to the file
            #         with open(file_path, 'w') as file:
            #             file.write(file_content)

            #     except Exception as e:
            #         print(f"Error processing {file_name} for 'fill-mask' task: {str(e)}")
            
            inputs.append(file_input)
    
    # Create an Input object for the folder containing all files
    folder_input = Input(path=folder_path, type=AssetTypes.URI_FOLDER)
    job_inputs = [folder_input] + inputs
    # print("job_inputs:", {job_inputs})
    return folder_path
    




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

    
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)

def create_or_get_compute_target(ml_client,  compute):
    cpu_compute_target = compute
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        print("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size=compute, min_instances=0, max_instances=3, idle_time_before_scale_down = 120
        )
        ml_client.compute.begin_create_or_update(compute).result()
    return compute

def get_latest_model_version(workspace_ml_client, test_model_name):
    print("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(test_model_name ))
    
    if len(version_list) == 0:
        print("Model not found in registry")
        foundation_model_name = None  # Set to None if the model is not found
        foundation_model_id = None  # Set id to None as well
    else:
        model_version = version_list[0].version
        foundation_model = workspace_ml_client.models.get(
            test_model_name , model_version)
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
    return foundation_model, foundation_model_name



def create_and_configure_batch_endpoint(
    foundation_model_name, foundation_model, compute, workspace_ml_client, task
):

    # Create a unique endpoint name using a timestamp
    timestamp = int(time.time())
    endpoint_name = task + str(timestamp)

    reserve_keywords = ["microsoft"]
    regx_for_reserve_keyword = re.compile(
        '|'.join(map(re.escape, reserve_keywords)))
    reserve_keywords_check = re.findall(
        regx_for_reserve_keyword, foundation_model_name)
    if reserve_keywords_check:
        foundation_model_name = regx_for_reserve_keyword.sub(
            '', foundation_model_name)
        foundation_model_name = foundation_model_name.lstrip("-")


    if foundation_model_name[0].isdigit():
            num_pattern = "[0-9]"
            foundation_model_name = re.sub(num_pattern, '', foundation_model_name)
            foundation_model_name = foundation_model_name.strip("-")
        # Check the model name is more then 32 character
    if len(foundation_model_name) > 32:
        model_name = foundation_model_name[:31]
        deployment_name = model_name.rstrip("-")
    else:
        deployment_name = foundation_model_name
            
            #endpoint_name = f"{registered_model_name}"
    print("Endpoint name:", {endpoint_name})
    print("Deployment name:", {deployment_name})

    # Create the BatchEndpoint
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description=f"Batch endpoint for {foundation_model.name} ",
    )
    workspace_ml_client.begin_create_or_update(endpoint).result()

    deployment_name = f"{deployment_name}"
    print("DN :", {deployment_name})

    # Create the BatchDeployment
    deployment = BatchDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=foundation_model.id,
        compute=compute,
        error_threshold=0,
        instance_count=1,
        logging_level="info",
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    )
    workspace_ml_client.begin_create_or_update(deployment).result()

    # Retrieve the created endpoint
    endpoint = workspace_ml_client.batch_endpoints.get(endpoint_name)

    # Set the default deployment name
    endpoint.defaults.deployment_name = deployment_name
    workspace_ml_client.begin_create_or_update(endpoint).wait()

    # Retrieve and print the default deployment name
    endpoint = workspace_ml_client.batch_endpoints.get(endpoint_name)
    print(f"The default deployment is {endpoint.defaults.deployment_name}")
    return endpoint_name




if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        # logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    # print values of all above variables
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: {queue['registry']}")
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
    
    compute_target = create_or_get_compute_target(workspace_ml_client, queue.compute)
    # environment_variables = {"test_model_name": test_model_name}
    env_list = workspace_ml_client.environments.list(name=queue.environment)
    latest_version = 0
    for env in env_list:
        if latest_version <= int(env.version):
            latest_version = int(env.version)
    print("Latest Environment Version:", latest_version)
    latest_env = workspace_ml_client.environments.get(
        name=queue.environment, version=str(latest_version))
    print("Latest Environment :", latest_env)
    
    #version_list = list(workspace_ml_client.models.list(test_model_name))
    client = MlflowClient()



    # task = foundation_model.flavors["transformers"]["task"]
    # print("task :", {task})
    # folder_path = get_task_specified_input(task=task, test_model_name=test_model_name)
    
    expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                            "$", "%", "^", "&", "*", "<", ">", "?", "!", "~"]
    # Create the regular expression to ignore
    regx_for_expression = re.compile(
        '|'.join(map(re.escape, expression_to_ignore)))
    # Check the model_name contains any of there character
    expression_check = re.findall(regx_for_expression, test_model_name)
    if expression_check:
        # Replace the expression with hyphen
        test_model_name  = regx_for_expression.sub("-", test_model_name)
        # print("test model name before lower - :", {test_model_name})
        # test_model_name  = test_model_name.lower()


   
    print("model name replaced with - :", {test_model_name})
    
    foundation_model, foundation_model_name = get_latest_model_version(workspace_ml_client, test_model_name.lower())
    

    task = foundation_model.flavors["transformers"]["task"]
    print("task :", {task})
    endpoint_name = create_and_configure_batch_endpoint(foundation_model_name, foundation_model, queue.compute, workspace_ml_client, task)
    
    folder_path = get_task_specified_input(task=task, test_model_name=test_model_name)

    #folder_path = get_task_specified_input(task=task, test_model_name)
    print(" input taken, running Batch Job")
    input = Input(path=folder_path, type=AssetTypes.URI_FOLDER)
    print("Invoking the job")

    # Invoke the batch endpoint
    job = workspace_ml_client.batch_endpoints.invoke(
        endpoint_name=endpoint_name, input=input
    )
    workspace_ml_client.jobs.stream(job.name)

    workspace_ml_client.batch_endpoints.begin_delete(name=endpoint_name).result()


