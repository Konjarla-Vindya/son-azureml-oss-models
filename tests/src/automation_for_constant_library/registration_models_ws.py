import json
import os
import threading
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import mlflow.azureml
from mlflow.azureml import Environment, Estimator
from azureml.core import Workspace

# Load configuration from the JSON file
with open("config_registration.json", "r") as config_file:
    config = json.load(config_file)

# Function to register a model in a workspace
def register_model(subscription_id, resource_group, workspace_name, model_name, model_path):
    try:
        # Acquire Azure credentials
        try:
            # Try to use DefaultAzureCredential for authentication
            credential = DefaultAzureCredential()
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential doesn't work
            credential = InteractiveBrowserCredential()

        workspace = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, auth=credential)

        registered_model = mlflow.register_model(model_path, model_name, workspace_name)
        print(f"Registered {model_name} in {workspace_name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace_name}: {str(e)}")

# Create and start threads for registration
threads = []
workspace_names = config["workspace_names"]
workspace_count = len(workspace_names)
model_names = config["model_names"]

# Initialize a set to keep track of registered models
registered_models = set()

for i, model_name in enumerate(model_names):
    workspace_name = workspace_names[i % workspace_count]
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    model_path = os.path.join(config["models_directory"], model_name)
    thread = threading.Thread(target=register_model, args=(subscription_id, resource_group, workspace_name, model_name, model_path))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
