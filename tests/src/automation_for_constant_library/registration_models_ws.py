import json
from azureml.core import Workspace, Model
import os
import threading
from huggingface_hub import HfApi
import pandas as pd
import re

# Load configuration from the JSON file
with open("config_registration.json", "r") as config_file:
    config = json.load(config_file)

# Dictionary to keep track of which model is registered in which workspace
registered_models = {}

# Lock for thread synchronization
lock = threading.Lock()

# Function to register a model in a workspace
def register_model(workspace_name, model_name, model_path):
    try:
        workspace = Workspace.get(name=workspace_name, subscription_id=config["subscription_id"], resource_group=config["resource_group"])
        # Check if the model has already been registered
        with lock:
            if model_name in registered_models:
                print(f"Model {model_name} is already registered in {registered_models[model_name]}. Skipping registration in {workspace_name}.")
                return
            model = Model.register(workspace, model_name, model_path)
            registered_models[model_name] = workspace_name
            print(f"Registered {model.name} in {workspace_name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace_name}: {str(e)}")

# Create and start threads for registration
threads = []
for workspace_name in config["workspace_names"]:
    for model_name in config["model_names"]:
        model_path = os.path.join(config["models_directory"], model_name)
        thread = threading.Thread(target=register_model, args=(workspace_name, model_name, model_path))
        thread.start()
        threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
