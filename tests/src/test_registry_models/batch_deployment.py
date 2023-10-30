from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

from utils.logging import get_logger
from transformers import AutoTokenizer
import time
import re
import sys
import os

logger = get_logger(__name__)

class ModelBatchDeployment:
    def __init__(self, model, workspace_ml_client, task, model_name) -> None:
        self.model = model
        self.workspace_ml_client = workspace_ml_client
        self.task = task
        self.model_name = model_name

    def get_deployemnt_endpoint_and_name(self):
        timestamp = int(time.time())
        endpoint_name = self.task + str(timestamp)
        # Expression need to be replaced with hyphen
        expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                                "$", "%", "^", "&", "*", "<", ">", "?", "!", "~", "_"]
        # Create the regular expression to ignore
        regx_for_expression = re.compile(
            '|'.join(map(re.escape, expression_to_ignore)))
        # Check the model_name contains any of there character
        expression_check = re.findall(regx_for_expression, self.model_name)
        latest_model_name = self.model_name
        
        if expression_check:
            # Replace the expression with hyphen
            latest_model_name = regx_for_expression.sub("-", self.model_name)
        # Reserve Keyword need to be removed
        reserve_keywords = ["microsoft"]
        # Create the regular expression to ignore
        regx_for_reserve_keyword = re.compile(
            '|'.join(map(re.escape, reserve_keywords)))
        # Check the model_name contains any of the string
        reserve_keywords_check = re.findall(
            regx_for_reserve_keyword, self.model_name)
        if reserve_keywords_check:
            # Replace the resenve keyword with empty
            latest_model_name = regx_for_reserve_keyword.sub(
                '', latest_model_name)
            latest_model_name = latest_model_name.lstrip("-")
        # Check if the model name starts with a digit
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
        # Check the model name is more then 32 character
        if len(latest_model_name) > 32:
            model_name = latest_model_name[:31]
            deployment_name = model_name.rstrip("-")
        else:
            deployment_name = latest_model_name
        return endpoint_name, deployment_name.lower()
        
        
    def get_batch_endpoint(self, endpoint_name):
        logger.info("Creating the batch endpoint...")
        model = self.model
        endpoint = BatchEndpoint(
                name=endpoint_name,
                description=f"Batch endpoint for {model.name} ",
            )
        try:
             self.workspace_ml_client.begin_create_or_update(endpoint).result()
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create batch end point\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            sys.exit(1)

    def create_batch_deployment(self, deployment_name, endpoint_name, compute):
        logger.info("Creating the batch deployment...")
        try:
            foundation_model = self.model
            # Create the BatchDeployment
            deployment = BatchDeployment(
                name=deployment_name,
                endpoint_name=endpoint_name,
                model=foundation_model.id,
                compute=compute,
                error_threshold=0,
                instance_count=1,
                logging_level="info",
                max_concurrency_per_instance=2,
                mini_batch_size=10,
                output_file_name="predictions.csv",
                retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
            )
            self.workspace_ml_client.begin_create_or_update(deployment).result()

            # Retrieve the created endpoint
            endpoint = self.workspace_ml_client.batch_endpoints.get(endpoint_name)

            # Set the default deployment name
            endpoint.defaults.deployment_name = deployment_name
            self.workspace_ml_client.begin_create_or_update(endpoint).wait()

            # Retrieve and print the default deployment name
            endpoint = self.workspace_ml_client.batch_endpoints.get(endpoint_name)
            print(f"The default deployment is {endpoint.defaults.deployment_name}")
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"::error:: Could not create batch deployment\n")
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {e}")
            sys.exit(1)

    def process_input_for_fill_mask_task(self, file_path, mask_token):
        try:
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Detect and replace the masking token based on the model's mask token
            file_content = re.sub(r'\[MASK\]', mask_token, file_content)
            #file_content = re.sub(r'<mask>', mask_token, file_content)

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write(file_content)

        except Exception as e:
            print(f"Error processing {file_path} for 'fill-mask' task: {str(e)}")

    def get_task_specified_input(self):
        logger.info("pulling inputs")
        folder_path = f"sample_inputs/{self.task}/batch_inputs"

        # List all file names in the folder
        file_names = os.listdir(folder_path)
        
        # Create a list to store individual input objects for each file
        inputs = []
        
        # Process each file in the folder
        for file_name in file_names:
            logger.info(f"File Name: {file_name}")
            # Construct the full path to the file
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Create an Input object for the file and add it to the list of inputs
                file_input = Input(path=file_path, type=AssetTypes.URI_FILE)
                # Handle the "fill-mask" task by replacing [MASK] with <mask> in the input data
                if self.task.lower() == "fill-mask":
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    #tokenizer = AutoTokenizer.from_pretrained(test_model_name, trust_remote_code=True, use_auth_token=True)
                    mask_token = tokenizer.mask_token  
                    self.process_input_for_fill_mask_task(file_path, mask_token)
                # if task.lower() == "fill-mask":
                #     try:
                #         with open(file_path, 'r') as file:
                #             file_content = file.read()
                        
                #         # Replace [MASK] with <mask> in the input data
                #         file_content = file_content.replace('[MASK]', '<mask>')
                        
                #         # Write the modified content back to the file
                #         with open(file_path, 'w') as file:
                #             file.write(file_content)

                #     except Exception as e:
                #         print(f"Error processing {file_name} for 'fill-mask' task: {str(e)}")
                
                inputs.append(file_input)
        
        # Create an Input object for the folder containing all files
        folder_input = Input(path=folder_path, type=AssetTypes.URI_FOLDER)
        job_inputs = [folder_input] + inputs
        # print("job_inputs:", {job_inputs})
        return folder_path
    def delete_endpoint(self, endpoint_name):
        logger.info("Deleting the endpoint...")
        try:
            self.workspace_ml_client.batch_endpoints.begin_delete(name=endpoint_name).result()
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : {ex}")
            logger.error(f"::warning:: Could not delete endpoint: : \n{ex}")
            exit(0)

    def batch_deployment(self, compute):
        endpoint_name, deployment_name = self.get_deployemnt_endpoint_and_name()
        self.get_batch_endpoint(endpoint_name=endpoint_name)
        self.create_batch_deployment(deployment_name=deployment_name, endpoint_name=endpoint_name, compute=compute)
        folder_path = self.get_task_specified_input()
        input = Input(path=folder_path, type=AssetTypes.URI_FOLDER)
        # Invoke the batch endpoint
        job = self.workspace_ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint_name, input=input
        )
        self.workspace_ml_client.jobs.stream(job.name)
        self.delete_endpoint(endpoint_name=endpoint_name)
        