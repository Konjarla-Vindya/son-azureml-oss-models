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
# from utils.logging import get_logger
# from fetch_task import HfTask
import mlflow
# from box import ConfigBox
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



# logger = get_logger(__name__)
# class BatchInferenceAndDeployemnt:
#     def __init__(self, test_model_name, workspace_ml_client, registry) -> None:
#         self.test_model_name = test_model_name
#         self.workspace_ml_client = workspace_ml_client
#         self.registry = registry

#     def get_error_messages(self):
#         # load ../../config/errors.json into a dictionary
#         with open('../../config/errors.json') as f:
#             return json.load(f)


    # def prase_logs(self, logs):
    #     error_messages = self.get_error_messages()
    #     # split logs by \n
    #     logs_list = logs.split("\n")
    #     # loop through each line in logs_list
    #     for line in logs_list:
    #         # loop through each error in errors
    #         for error in error_messages:
    #             # if error is found in line, print error message
    #             if error['parse_string'] in line:
    #                 # logger.error(
    #                 #     f"::error:: {error_messages['error_category']}: {line}")

def create_or_update_batch_endpoint(workspace_ml_client, model_detail, description=""):
    # Generate a unique endpoint name based on the current timestamp
    timestamp = int(time.time())
    endpoint_name = f"fill-maskws-{timestamp}"

    # Create or update the Batch Endpoint
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description=f"Batch endpoint for {model_detail.name}, for fill-mask task{description}",
    )
    workspace_ml_client.begin_create_or_update(endpoint).result()
    return endpoint

    # Example usage:
    # Replace the parameters with your desired values
    model_detail = model_detail.id  # Provide your foundation model object
    description = "By Automation"

def create_or_update_batch_deployment(
    workspace_ml_client,
    deployment_name,
    endpoint_name,
    model_detail,
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
        model=model_detail.id,
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
    deployment_name = "demo"
    endpoint_name = "fill-mask-Auto"  # Provide the actual endpoint name
    model_detail = model_detail.id  # Provide your foundation model object
    compute = "queue.compute"  # Provide the compute name
    #test_model_name
    created_deployment = create_or_update_batch_deployment(
        workspace_ml_client,
        deployment_name,
        endpoint_name,
        model_detail,
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
    created_endpoint = create_or_update_batch_endpoint(workspace_ml_client, foundation_model, description)
    # Example usage:
    # Replace the parameters with your desired values
    #endpoint_name = "your_endpoint_name"  # Provide the actual endpoint name
    deployment_name = "demo-Auto"  # Provide the new default deployment name

    # Example usage:
    # Replace the parameters with your desired values
    workspace = queue.workspace  # Provide your Azure ML workspace object
    #endpoint_name = "your_endpoint_name"  # Provide the actual endpoint name
    #batch_inputs_dir = "your_datastore_path"  # Provide the path to the input data in the datastore

    #invoke_batch_job(workspace, endpoint_name, batch_inputs_dir)
