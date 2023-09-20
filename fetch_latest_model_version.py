%pip install --upgrade azureml-sdk
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Model
import json

# Specify the path to your configuration JSON file
config_file_path = "son-azureml-oss-models/config.json"

def fetch_latest_model_version():
    # Load Azure ML workspace using the default credential
    credential = DefaultAzureCredential()
    ws = Workspace.from_config(path=config_file_path, auth=credential)

    # Extract parameters from the configuration JSON
    subscription_id = ws.subscription_id
    resource_group = ws.resource_group
    workspace_name = ws.name
    model_name = ws.model_name

    # Fetch the latest registered model version
    latest_model_version = None
    latest_version = 0

    for model in Model.list(workspace=ws, name=model_name):
        if model.version > latest_version:
            latest_version = model.version
            latest_model_version = model.version

    if latest_model_version is not None:
        # Save the latest model version to a file
        with open("latest_model_version.txt", "w") as version_file:
            version_file.write(json.dumps({"latest_model_version": latest_model_version}))
    else:
        print("No model versions found.")

    print(f"Latest model version: {latest_model_version}")

if __name__ == "__main__":
    fetch_latest_model_version()
