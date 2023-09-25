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
from utils.logging import get_logger
import mlflow
from box import ConfigBox
import re
import sys

logger = get_logger(__name__)


class OnlineDeployment:
    def __init__(self, workspace_ml_client, latest_model) -> None:
        self.workspace_ml_client = workspace_ml_client
        self.latest_model = latest_model

    def get_task_specified_input(self, task):
        #scoring_file = f"../../config/sample_inputs/{self.registry}/{task}.json"
        scoring_file = f"sample_inputs/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
                logger.info(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            logger.warning(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")
        return scoring_file, scoring_input

    def create_online_endpoint(self, endpoint):
        logger.info("In create_online_endpoint...")
        try:
            self.workspace_ml_client.online_endpoints.begin_create_or_update(
                endpoint).wait()
        except Exception as e:
            logger.error(f"::error:: Could not create endpoint: \n")
            logger.error(f"{e}\n\n check logs:\n\n")
            self.prase_logs(str(e))
            exit(1)
        online_endpoint_obj = self.workspace_ml_client.online_endpoints.get(
            name=endpoint.name)
        logger.info(f"online_endpoint_obj : {online_endpoint_obj}")

    def create_online_deployment(self, latest_model, online_endpoint_name, instance_type, endpoint):
        logger.info("In create_online_deployment...")
        logger.info(f"latest_model.name is this : {latest_model.name}")
        latest_model_name = self.get_model_name(
            latest_model_name=latest_model.name)
        # Check if the model name starts with a digit
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
        # Check the model name is more then 32 character
        if len(latest_model.name) > 32:
            model_name = latest_model_name[:31]
            deployment_name = model_name.rstrip("-")
        else:
            deployment_name = latest_model_name
        logger.info(f"deployment name is this one : {deployment_name}")
        deployment_config = ManagedOnlineDeployment(
            name=deployment_name,
            model=latest_model.id,
            endpoint_name=online_endpoint_name,
            instance_type=instance_type,
            instance_count=1,
            request_settings=OnlineRequestSettings(
                max_concurrent_requests_per_instance=1,
                request_timeout_ms=50000,
                max_queue_wait_ms=500,
            )
        )
        try:
            self.workspace_ml_client.online_deployments.begin_create_or_update(
                deployment_config).wait()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create deployment\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            self.prase_logs(str(e))
            self.get_online_endpoint_logs(
                deployment_name, online_endpoint_name)
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
            exit(1)
        endpoint.traffic = {deployment_name: 100}
        try:
            self.workspace_ml_client.begin_create_or_update(endpoint).result()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create deployment\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            self.get_online_endpoint_logs(
                deployment_name, online_endpoint_name)
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=endpoint.name).wait()
            exit(1)
        deployment_obj = self.workspace_ml_client.online_deployments.get(
            name=deployment_name, endpoint_name=endpoint.name)
        logger.info(f"Deployment object is this one: {deployment_obj}")
        return deployment_name
    
    def cloud_inference(self, scoring_file, scoring_input, online_endpoint_name, deployment_name):
        try:
            logger.info(f"endpoint_name : {online_endpoint_name}")
            logger.info(f"deployment_name : {deployment_name}")
            logger.info(f"Input data is this one : {scoring_input}")
            response = self.workspace_ml_client.online_endpoints.invoke(
                endpoint_name=online_endpoint_name,
                deployment_name=deployment_name,
                request_file=scoring_file,
            )
            response_json = json.loads(response)
            output = json.dumps(response_json, indent=2)
            logger.info(f"response: \n\n{output}")
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fh:
                print(f'####Sample input', file=fh)
                print(f'```json', file=fh)
                print(f'{scoring_input}', file=fh)
                print(f'```', file=fh)
                print(f'####Sample output', file=fh)
                print(f'```json', file=fh)
                print(f'{output}', file=fh)
                print(f'```', file=fh)
        except Exception as e:
            logger.error(f"::error:: Could not invoke endpoint: \n")
            logger.info(f"{e}\n\n check logs:\n\n")

    def delete_online_endpoint(self, online_endpoint_name):
        try:
            logger.info("\n In delete_online_endpoint.....")
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            logger.error(f"::warning:: Could not delete endpoint: : \n{e}")
            exit(0)
    
    def model_online_deployment(self, instance_type):
        logger.info("Started model online deployment")
        task = self.latest_model.tags["task"]
        logger.info(f"Task is : {task}")
        scoring_file, scoring_input = self.get_task_specified_input(task=task)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        timestamp = int(time.time())
        online_endpoint_name = task + str(timestamp)
        #online_endpoint_name = "Testing" + str(timestamp)
        logger.info(f"online_endpoint_name: {online_endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=online_endpoint_name,
            auth_mode="key",
        )
        self.create_online_endpoint(endpoint=endpoint)
        deployment_name = self.create_online_deployment(
            latest_model=self.latest_model,
            online_endpoint_name=online_endpoint_name,
            instance_type=instance_type,
            endpoint=endpoint
        )
        self.cloud_inference(
            scoring_file=scoring_file,
            scoring_input=scoring_input,
            online_endpoint_name=online_endpoint_name,
            deployment_name=deployment_name
        )
        self.delete_online_endpoint(online_endpoint_name=online_endpoint_name)
        logger.info("Model online deployment execution completed successfully")
