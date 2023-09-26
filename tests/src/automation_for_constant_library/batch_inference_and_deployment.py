import time
import json
import os
from azureml.core import Run
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
    Model,
    ModelConfiguration,
    ModelPackage,
    AzureMLOnlineInferencingServer
)
import mlflow
from box import ConfigBox
import re
import sys
import time
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    Model,
)
from azureml.core.datastore import Datastore
from azureml.core import Workspace
from mlflow.tracking.client import MlflowClient

# Import the BatchDeployemnt class you defined
# from batch_inference_and_deployment import BatchDeployemnt
class BatchDeployment:
    def __init__(self, test_model_name, workspace_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry = registry

def create_and_configure_batch_endpoint(
    foundation_model, compute_name, workspace_ml_client
):
    # Create a unique endpoint name using a timestamp
    timestamp = int(time.time())
    endpoint_name = f"fill-maskws1-{timestamp}"

    # Create the BatchEndpoint
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description=f"Batch endpoint for {foundation_model.name}, for fill-mask task",
    )
    workspace_ml_client.begin_create_or_update(endpoint).result()

def get_latest_model_version(workspace_ml_client, test_model_name):
        print("In get_latest_model_version...")
        version_list = list(workspace_ml_client.models.list(test_model_name))
        
        if len(version_list) == 0:
            print("Model not found in registry")
            foundation_model_name = None  # Set to None if the model is not found
            foundation_model_id = None  # Set id to None as well
        else:
            model_version = version_list[0].version
            foundation_model = workspace_ml_client.models.get(
                test_model_name, model_version)
            print(
                "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                    foundation_model.name, foundation_model.version, foundation_model.id
                )
            )
            foundation_model_name = foundation_model.name  # Assign the value to a new variable
            foundation_model_id = foundation_model.id  # Assign the id to a new variable
        
        # Check if foundation_model_name and foundation_model_id are None or have values
        if foundation_model_name and foundation_model_id:
            print(f"Latest model {foundation_model_name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
            print("foundation_model.name:", foundation_model_name)
            print("foundation_model.id:", foundation_model_id)
        else:
            print("No model found in the registry.")
        
        #print(f"Model Config : {latest_model.config}")
        return foundation_model

def main():
    # Initialize the MLflow client
    mlflow_client = MlflowClient()

    # # Replace these variables with your actual values
    # test_model_name = "your_test_model_name"
    # registry = "your_registry"
    # foundation_model_id = "your_foundation_model_id"
    # compute_name = "your_compute_name"
    # workspace = "your_workspace"

    # Create a BatchDeployment instance
    batch_deployment = BatchDeployment(
        test_model_name=test_model_name,
        workspace_ml_client=mlflow_client,  # Use the MLflow client
        registry=registry,
    )

    # Get the latest model version
    foundation_model = batch_deployment.get_latest_model_version(
        mlflow_client, test_model_name
    )

    # Create and configure the Batch Endpoint
    created_endpoint = create_and_configure_batch_endpoint(
        foundation_model, compute_name, mlflow_client
    )

    # Create and configure the Batch Deployment
    created_deployment = batch_deployment.create_or_update_batch_deployment(
        deployment_name="demo",
        endpoint_name=created_endpoint.name,
        foundation_model=foundation_model,
        compute=compute_name,
    )

    # Set the default Batch Deployment
    batch_deployment.set_default_batch_deployment(
        mlflow_client, created_endpoint.name, created_deployment.name
    )

if __name__ == "__main__":
    main()
