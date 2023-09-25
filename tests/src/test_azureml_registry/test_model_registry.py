from azureml.core import Workspace, Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
#from azure.ai.ml.entities import AmlCompute
#from azure.ai.ml import command
from azure.ai.ml import MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
from typing import *
from test_online_deployment import OnlineDeployment

logger = get_logger(__name__)

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

TASK_NAME = ['fill-mask', 'token-classification', 'question-answering',
             'summarization', 'text-generation', 'text-classification', 'text-translation']


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))


def get_model_list(registry_mlclient) -> List:
    registered_model_list = []
    temp_list = registry_mlclient.models.list()
    for model in temp_list:
        registered_model_list.append(model.name)
    return registered_model_list


def get_latest_model_version(registry_mlclient, model_list):
    for model_name in model_list:
        model_versions = registry_mlclient.models.list(model_name)
        model_version_count = 0
        models = []
        for model in model_versions:
            model_version_count = model_version_count + 1
            models.append(model)
        # Sort models by creation time and find the latest model
        sorted_models = sorted(
            models, key=lambda x: x.creation_context.created_at, reverse=True)
        latest_model = sorted_models[0]
        if latest_model.tags.get("task") != None:
            task = latest_model.tags["task"]
            if task in TASK_NAME:
                logger.info(
                    f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}")
                return latest_model


def main():
    queue = get_test_queue()
    # print values of all above variables
    logger.info(f"test_subscription_id: {queue['subscription']}")
    logger.info(f"test_resource_group: {queue['subscription']}")
    logger.info(f"test_workspace_name: {queue['workspace']}")
    #logger.info (f"test_registry: {queue['registry']}")
    logger.info(f"test_queue: {test_queue}")
    logger.info(f"test_set: {test_set}")
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
    registry_mlclient = MLClient(
        credential=credential,
        registry_name="azureml"
    )
    registered_model_list = get_model_list(registry_mlclient=registry_mlclient)
    latest_model = get_latest_model_version(
        registry_mlclient=registry_mlclient, model_list=registered_model_list)

    onlineDeployment = OnlineDeployment(
        workspace_ml_client=workspace_ml_client, latest_model=latest_model)
    onlineDeployment.model_online_deployment(instance_type=queue.instance_type)


if __name__ == "__main__":
    main()
