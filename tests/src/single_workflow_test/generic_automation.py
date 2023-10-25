from azureml.core import Workspace, Environment
from model_inference_and_deployment import ModelInferenceAndDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import MLClient, UserIdentityConfiguration
import mlflow
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
from fetch_task import HfTask
from azure.ai.ml.dsl import pipeline
from azure.core.exceptions import ResourceNotFoundError
import time
import re
import threading

# constants
# check_override = True
# huggingface_model_exists_in_registry = False

logger = get_logger(__name__)

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



# finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command

def multi_thread_deployment(workspace_ml_client, model_name, task):
    InferenceAndDeployment = ModelInferenceAndDeployemnt(
        test_model_name=model_name,
        workspace_ml_client=workspace_ml_client,
        registry=queue.registry
    )
    InferenceAndDeployment.model_infernce_and_deployment(
        instance_type=queue.instance_type,
        compute=queue.compute,
        task = task
    )

if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    # print values of all above variables
    logger.info (f"test_subscription_id: {queue['subscription']}")
    logger.info (f"test_resource_group: {queue['subscription']}")
    logger.info (f"test_workspace_name: {queue['workspace']}")
    logger.info (f"test_model_name: {test_model_name}")
    logger.info (f"test_sku_type: {test_sku_type}")
    logger.info (f"test_trigger_next_model: {test_trigger_next_model}")
    logger.info (f"test_queue: {test_queue}")
    logger.info (f"test_set: {test_set}")
    logger.info(f"Here is my test model name : {test_model_name}")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    logger.info(f"workspace_name : {queue.workspace}")
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
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
    registry_ml_client = MLClient(
        credential=credential,
        registry_name=queue.registry
    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    model_list = list(queue.models)
    model_df = HfTask().get_task()
    thread_list = []
    for model_name in model_list:
        required_data = model_df[model_df.modelId.apply(lambda x: x == model_name)]
        required_data = required_data["pipeline_tag"].to_string()
        pattern = r'[0-9\s+]'
        task = re.sub(pattern, '', required_data)
        timestamp = int(time.time())
        thread = f"{model_name}-timestamp"
        thread = threading.Thread(target=multi_thread_deployment, args=(workspace_ml_client, model_name, task))
        thread.start()
        thread_list.append(thread)
    for thred in thread_list:
        thread.join()
        # InferenceAndDeployment = ModelInferenceAndDeployemnt(
        #     test_model_name=test_model_name,
        #     workspace_ml_client=workspace_ml_client,
        #     registry=queue.registry
        # )
        # InferenceAndDeployment.model_infernce_and_deployment(
        #     instance_type=queue.instance_type,
        #     compute=queue.compute,
        #     task = task
        # )
