from azureml.core import Workspace, Environment
from model_inference_and_deployment import ModelInferenceAndDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
# from utils.logging import get_logger
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from mlflow.tracking.client import MlflowClient

test_model_name = os.environ.get('test_model_name')

class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

def get_latest_model_version(self, workspace_ml_client, model_name):
    print("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(model_name))
    if len(version_list) == 0:
        print("Model not found in registry")
    else:
        model_version = version_list[0].version
        foundation_model = workspace_ml_client.models.get(
            model_name, model_version)
        print(
            "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                foundation_model.name, foundation_model.version, foundation_model.id
            )
        )
    print(
        f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
    print(f"Model Config : {latest_model.config}")
    return foundation_model



if __name__ == "__main__":
    model = Model(model_name=test_model_name)
    # Get the sample input data
    task = model.get_task()
    # Get the sample input data
    scoring_input = model.get_sample_input_data(task=task)
    print("This is the task associated to the model : ", task)
    # If threr will be model namr with / then replace it
    registered_model_name = test_model_name.replace("/", "-")
    client = MlflowClient()
    model.download_and_register_model(
        task=task, scoring_input=scoring_input, registered_model_name=registered_model_name, client=client)
    model.registered_model_inference(
        task=task, scoring_input=scoring_input, registered_model_name=registered_model_name, client=client)