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

class ModelInferenceAndDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry = registry

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
        return foundation_model

    def sample_inference(self, latest_model, registry, workspace_ml_client, online_endpoint_name):
        # get the task tag from the latest_model.tags
        tags = str(latest_model.tags)
        # replace single quotes with double quotes in tags
        tags = tags.replace("'", '"')
        # convert tags to dictionary
        tags_dict = json.loads(tags)
        task = tags_dict['task']
        print(f"task: {task}")
        scoring_file = f"../../config/sample_inputs/{registry}/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = json.load(f)
                print(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")

        # invoke the endpoint
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=online_endpoint_name,
                deployment_name="demo",
                request_file=scoring_file,
            )
            response_json = json.loads(response)
            output = json.dumps(response_json, indent=2)
            print(f"response: \n\n{output}")
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
            print(f"::error:: Could not invoke endpoint: \n")
            print(f"{e}\n\n check logs:\n\n")

    def create_model_package(self, latest_model, endpoint):
        print("In create_model_package...")
        model_configuration = ModelConfiguration(mode="download")
        package_name = f"package-v2-{latest_model.name}"
        package_config = ModelPackage(
            target_environment_name=package_name,
            inferencing_server=AzureMLOnlineInferencingServer(),
            model_configuration=model_configuration
        )
        model_package = self.workspace_ml_client.models.package(
            latest_model.name,
            latest_model.version,
            package_config
        )
        self.workspace_ml_client.begin_create_or_update(endpoint).result()
        return model_package

    def create_online_deployment(self, latest_model, online_endpoint_name, model_package, instance_type):
        print("In create_online_deployment...")
        print("latest_model.name is this : ", latest_model.name)
        #Expression need to be replaced with hyphen
        expression_to_ignore = ["/","\\", "|", "@", "#", ".", "$", "%", "^", "&", "*", "<", ">", "?", "!", "~", "_"]
        #Create the regular expression to ignore
        regx = re.compile('|'.join(map(re.escape, expression_to_ignore)))
        # Check the model_name contains any of there character
        expression_check = re.findall(regx, latest_model.name)
        if expression_check:
            #Replace the expression with hyphen
            latest_model_name = regx.sub("-", latest_model.name)
        else:
            latest_model_name = latest_model.name

        #Check if the model name starts with a digit
        if latest_model_name[0].isdigit():
            num_pattern = "[0-9]"
            latest_model_name = re.sub(num_pattern, '', latest_model_name)
            latest_model_name = latest_model_name.strip("-")
        #Check the model name is more then 32 character
        if len(latest_model.name) > 32:
            model_name = latest_model_name[:31]
            deployment_name = model_name.rstrip("-")
        else:
            deployment_name = latest_model_name
        print("deployment name is this one : ", deployment_name)
        deployment_config = ManagedOnlineDeployment(
            name=deployment_name,
            model=latest_model,
            endpoint_name=online_endpoint_name,
            environment=model_package,
            instance_type=instance_type,
            instance_count=1
        )
        deployment = self.workspace_ml_client.online_deployments.begin_create_or_update(
            deployment_config).result()

    def delete_online_endpoint(self, online_endpoint_name):
        try:
            print("\n In delete_online_endpoint.....")
            self.workspace_ml_client.online_endpoints.begin_delete(
                name=online_endpoint_name).wait()
        except Exception as e:
            print(f"::warning:: Could not delete endpoint: : \n{e}")
            exit(0)

    def model_infernce_and_deployment(self, instance_type):
        model_name = self.test_model_name.replace("/", "-")
        latest_model = self.get_latest_model_version(
            self.workspace_ml_client, model_name)
        task = latest_model.flavors["transformers"]["task"]
        print("latest_model:", latest_model)
        print("Task is : ", task)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        timestamp = int(time.time())
        online_endpoint_name = task + str(timestamp)
        print(f"online_endpoint_name: {online_endpoint_name}")

        endpoint = ManagedOnlineEndpoint(
            name=online_endpoint_name,
            auth_mode="key",
        )
        model_package = self.create_model_package(
            latest_model=latest_model, endpoint=endpoint)
        self.create_online_deployment(
            latest_model=latest_model,
            online_endpoint_name=online_endpoint_name,
            model_package=model_package,
            instance_type=instance_type
        )
        self.delete_online_endpoint(online_endpoint_name=online_endpoint_name)