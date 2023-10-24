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
import threading
from box import ConfigBox
from mlflow.tracking.client import MlflowClient



import json
from azureml.core import Workspace, Model
import os
import threading
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

# Function to fetch models from Hugging Face
def fetch_models_from_hf():
    hf_api = HfApi()
    models = hf_api.list_models()
    return [model.modelId for model in models]

# Create and start threads for registration
threads = []
workspace_names = config["workspace_names"]
workspace_count = len(workspace_names)
model_names = fetch_models_from_hf()

for i, model_name in enumerate(model_names):
    workspace_name = workspace_names[i % workspace_count]
    workspace = Workspace.get(name=workspace_name, subscription_id=config["subscription_id"], resource_group=config["resource_group"])
    model_path = os.path.join(config["models_directory"], model_name)
    thread = threading.Thread(target=register_model, args=(workspace, model_name, model_path))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
