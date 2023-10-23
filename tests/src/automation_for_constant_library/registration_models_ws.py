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

# Function to register a model in a workspace
def register_model(workspace, model_name, model_path):
    try:
        model = Model.register(workspace, model_name, model_path)
        print(f"Registered {model.name} in {workspace.name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace.name}: {str(e)}")

# Create and start threads for registration
threads = []
workspace_names = config["workspace_names"]
workspace_count = len(workspace_names)
model_names = config["model_names"]
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
