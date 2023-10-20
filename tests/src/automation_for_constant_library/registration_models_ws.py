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
def clean_model_name(model_name):
    # Remove any characters that are not letters, numbers, dashes, periods, or underscores
    cleaned_model_name = re.sub(r'[^a-zA-Z0-9-._/]', '_', model_name)
    # Ensure the model name starts with a letter or number
    if not cleaned_model_name[0].isalnum():
        cleaned_model_name = 'model_' + cleaned_model_name
    # Limit the model name to 255 characters
    return cleaned_model_name[:255]

def register_model(workspace_name, model_name, model_path):
    try:
        cleaned_model_name = clean_model_name(model_name)
        workspace = Workspace.get(name=workspace_name, subscription_id=config["subscription_id"], resource_group=config["resource_group"])
        model = Model.register(workspace, cleaned_model_name, model_path)
        print(f"Registered {model.name} in {workspace_name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace_name}: {str(e)}")

# def register_model(workspace_name, model_name, model_path):
#     try:
#         workspace = Workspace.get(name=workspace_name, subscription_id=config["subscription_id"], resource_group=config["resource_group"])
#         model = Model.register(workspace, model_name, model_path)
#         print(f"Registered {model.name} in {workspace_name}")
#     except Exception as e:
#         print(f"An error occurred while working with {workspace_name}: {str(e)}")

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
