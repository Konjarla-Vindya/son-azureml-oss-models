import time
import json
import os
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
# from azureml.core.webservice import BatchEndpoint
# from azureml.core.webservice import BatchDeployment
# from azureml.core.webservice import BatchEndpoint
# from azureml.core.webservice import BatchRetrySettings
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    Model,
)
from azureml.core.datastore import Datastore
# from azureml.data import InputDataType
from azureml.core import Workspace

def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
        
class BatchDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry = registry
        model_name = self.test_model_name
        latest_model = self.get_latest_model_version(self.workspace_ml_client, model_name)
        deployment_name = "Autodemo"
        compute = "queue.compute" 
        workspace = queue.workspace 

    def get_latest_model_version(self, workspace_ml_client, model_name):
        print("In get_latest_model_version...")
        version_list = list(workspace_ml_client.models.list(model_name))
        if len(version_list) == 0:
            print("Model not found in registry")
        else:
            model_version = version_list[0].version
            foundation_model = workspace_ml_client.models.get(
                model_name, model_version)
            print(
                "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                    foundation_model.name, foundation_model.version, foundation_model.id
                )
            )
        print(
            f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        print(f"Model Config : {latest_model.config}")
        return foundation_model


def create_or_update_batch_endpoint(workspace_ml_client, foundation_model, description=""):
    # Generate a unique endpoint name based on the current timestamp
    timestamp = int(time.time())
    endpoint_name = f"fill-maskwsauto-{timestamp}"

    # Create or update the Batch Endpoint
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description=f"Batch endpoint for {foundation_model.name}, for fill-mask task{description}",
    )
    workspace_ml_client.begin_create_or_update(endpoint).result()
    return endpoint

    # # Example usage:
    # # Replace the parameters with your desired values
    # foundation_model = model_detail.id  # Provide your foundation model object
    # description = "By Automation"

def create_or_update_batch_deployment(
    workspace_ml_client,
    deployment_name,
    endpoint_name,
    foundation_model,
    compute,
    error_threshold=0,
    instance_count=1,
    logging_level="info",
    max_concurrency_per_instance=2,
    mini_batch_size=10,
    output_file_name="predictions.csv",
    max_retries=3,
    timeout=300,
):
    deployment = BatchDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=foundation_model.id,
        compute=compute,
        error_threshold=error_threshold,
        instance_count=instance_count,
        logging_level=logging_level,
        max_concurrency_per_instance=max_concurrency_per_instance,
        mini_batch_size=mini_batch_size,
        output_file_name=output_file_name,
        retry_settings=BatchRetrySettings(max_retries=max_retries, timeout=timeout),
    )
    workspace_ml_client.begin_create_or_update(deployment).result()
    return deployment

def set_default_batch_deployment(workspace_ml_client, endpoint_name, deployment_name):
    # Get the existing Batch Endpoint
    endpoint = workspace_ml_client.batch_endpoints.get(endpoint_name)

    # Update the default deployment name
    endpoint.defaults.deployment_name = deployment_name

    # Save the updated endpoint
    workspace_ml_client.begin_create_or_update(endpoint).wait()

    # Retrieve and print the default deployment name
    updated_endpoint = workspace_ml_client.batch_endpoints.get(endpoint_name)
    print(f"The default deployment is {updated_endpoint.defaults.deployment_name}")



    # def invoke_batch_job(
    #     workspace: Workspace,
    #     endpoint_name: str,
    #     batch_inputs_dir: str,
    # ):
    #     # Get the Batch Endpoint
    #     endpoint = workspace.batch_endpoints[endpoint_name]

    #     # Define the input data
    #     input_data = Datastore(workspace=workspace, name=batch_inputs_dir)
    #     input = Input(input_data, input_type=InputDataType.MOUNT)

    #     # Invoke the batch job
    #     job = endpoint.invoke(input)

    #     # Stream job logs (optional)
    #     job.wait_for_completion()
    #     job_logs = workspace.jobs.stream(job.name)
    #     for log_line in job_logs:
    #         print(log_line)


if __name__ == "__main__":
    # Example usage:
    # Replace the parameters with your desired values
    # deployment_name = "demo"
    # endpoint_name = "fill-mask-Auto"  # Provide the actual endpoint name
    #foundation_model = foundation_model.id  # Provide your foundation model object
    # compute = "queue.compute"  # Provide the compute name
    #test_model_name
    test_model_name = os.environ.get('test_model_name')
    test_queue = os.environ.get('test_queue')
    test_set = os.environ.get('test_set')
    queue = get_test_queue()
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    print("workspace_name : ", queue.workspace)
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    
    BEDeployment = BatchDeployemnt(
            test_model_name=test_model_name,
            workspace_ml_client=workspace_ml_client,
            registry=queue.registry
        )
    created_endpoint = create_or_update_batch_endpoint(workspace_ml_client, foundation_model, description)
    created_deployment = create_or_update_batch_deployment(
        workspace_ml_client,
        deployment_name,
        endpoint_name,
        foundation_model,
        compute,
        error_threshold=0,
        instance_count=1,
        logging_level="info",
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        output_file_name="predictions.csv",
        max_retries=3,
        timeout=300,
    )
    set_default_batch_deployment(workspace_ml_client, endpoint_name, deployment_name)
    
    # Example usage:
    # Replace the parameters with your desired values
    #endpoint_name = "your_endpoint_name"  # Provide the actual endpoint name
    #deployment_name = "demo-Auto"  # Provide the new default deployment name

    # Example usage:
    # Replace the parameters with your desired values
    #workspace = queue.workspace  # Provide your Azure ML workspace object
    #endpoint_name = "your_endpoint_name"  # Provide the actual endpoint name
    #batch_inputs_dir = "your_datastore_path"  # Provide the path to the input data in the datastore

    #invoke_batch_job(workspace, endpoint_name, batch_inputs_dir)
