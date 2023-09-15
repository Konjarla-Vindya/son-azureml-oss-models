from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Model
import json

def fetch_latest_model_version(config):
    # Extract parameters from the configuration JSON
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace_name = config["workspace_name"]
    model_name = config["model_name"]

    # Load Azure ML workspace using the default credential
    credential = DefaultAzureCredential()

    ws = Workspace(subscription_id, resource_group, workspace_name, auth=credential)

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
    import argparse

    parser = argparse.ArgumentParser(description="Fetch the latest model version from Azure ML workspace.")
    parser.add_argument("--config_file", required=True, help="Path to the JSON configuration file")

    args = parser.parse_args()

    # Load the configuration from the JSON file
    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    fetch_latest_model_version(config)
