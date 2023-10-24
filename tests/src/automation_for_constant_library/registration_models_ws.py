# from azureml.core import Workspace, Environment
# from model_inference_and_deployment import ModelInferenceAndDeployemnt
# from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
# from azure.ai.ml.entities import AmlCompute
# from azure.ai.ml import command
# from azure.ai.ml import MLClient
# import mlflow
# import json
# import os
# import sys
# import threading
# from box import ConfigBox
# from mlflow.tracking.client import MlflowClient



import json
from azureml.core import Workspace, Model
import os
from huggingface_hub import HfApi

# Load configuration from the JSON file
with open("config_registration.json", "r") as config_file:
    config = json.load(config_file)

# Function to register a model in a workspace
def register_model(workspace, model_name, model_path):
    try:
        model = Model.register(workspace, model_name, model_path)
        print(f"Registered {model.name} in {workspace.name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace.name}: {str(e)}")

# Function to fetch a model from Hugging Face
def fetch_model_from_hf(model_name):
    hf_api = HfApi()
    model_info = hf_api.model_info(model_name)
    model_path = model_info.repo_id if model_info and 'repo_id' in model_info else None
    return model_path

# Fetch the model from Hugging Face
model_name = config["model_names"]
model_path = fetch_model_from_hf(model_name)

if model_path:
    # Register the model in the Azure ML workspace
    workspace_name = config["workspace_name"]
    workspace = Workspace.get(name=workspace_name, subscription_id=config["subscription_id"], resource_group=config["resource_group"])
    register_model(workspace, model_name, model_path)
else:
    print(f"Failed to fetch model path for {model_name}.")
