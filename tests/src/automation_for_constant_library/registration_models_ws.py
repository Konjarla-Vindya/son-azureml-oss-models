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



from azureml.core import Workspace, Environment, ComputeTarget
import mlflow
import os

# Load configuration from the JSON file
with open("config_registration.json", "r") as config_file:
    config = json.load(config_file)

# Connect to the Azure ML workspace
subscription_id = config["subscription_id"]
resource_group = config["resource_group"]
workspace_name = config["workspace_names"]
workspace = Workspace.get(subscription_id=subscription_id, resource_group=resource_group, name=workspace_name)

# Define an environment for your model
myenv = Environment(name="myenv")
myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=["scikit-learn"])

# Create or get a compute target
compute_name = "mycompute"
try:
    compute_target = ComputeTarget(workspace=workspace, name=compute_name)
except Exception as e:
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2", max_nodes=4)
    compute_target = ComputeTarget.create(workspace, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Function to register a model in a workspace
def register_model(subscription_id, resource_group, workspace_name, model_name, model_path):
    try:
        # Connect to the Azure ML workspace
        workspace = Workspace.get(subscription_id=subscription_id, resource_group=resource_group, name=workspace_name)

        # Set the environment for the model
        environment_name = myenv.name
        registered_environment = myenv.register(workspace)

        # Set the compute target for model registration
        compute_target_name = compute_target.name

        # Register the model with MLflow
        mlflow.register_model(
            model_uri=model_path,
            name=model_name,
            workspace=workspace,
            registered_env=registered_environment,
            compute_target=compute_target_name
        )

        print(f"Registered {model_name} in {workspace.name}")
    except Exception as e:
        print(f"An error occurred while working with {workspace_name}: {str(e)}")

# Create and start threads for registration
threads = []
workspace_names = config["workspace_names"]
workspace_count = len(workspace_names)
model_names = config["model_names"]

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
