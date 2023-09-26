from azureml.core import Workspace, Environment
from mlflow.tracking.client import MlflowClient
# from batch_inference_and_deployment import BatchDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import MLClient
import mlflow
import json
import os
import sys
# from box import ConfigBox
# from utils.logging import get_logger
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

# test_model_name = os.environ.get('test_model_name')

# class Model:
#     def __init__(self, model_name) -> None:
#         self.model_name = model_name



# def get_latest_model_version(self, client, model_name):
#     print("In get_latest_model_version...")
#     version_list = list(workspace_ml_client.models.list(model_name))
#     if len(version_list) == 0:
#         print("Model not found in registry")
#     else:
#         model_version = version_list[0].version
#         foundation_model = workspace_ml_client.models.get(
#             model_name, model_version)
#         print(
#             "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
#                 foundation_model.name, foundation_model.version, foundation_model.id
#             )
#         )
#     print(
#         f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
#     print(f"Model Config : {latest_model.config}")
#     return foundation_model
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



# if __name__ == "__main__":
#     model = Model(model_name=test_model_name)
#     print("Model name: " , model)
#     version_list = list(workspace_ml_client.models.list(test_model_name))
#     client = MlflowClient()
#     registered_model_detail = client.get_latest_versions(name=test_model_name, stages=["None"])
#     model_detail = registered_model_detail[0]
#     print("Latest model: ", model_detail)
#     print("Latest registered model: ", registered_model_detail)
#     print("Latest registered model version is : ", model_detail.version)
#     # print("Latest registered model id is : ", model_detail.id)
#     # print("Latest registered model name is : ", model_detail.name)

#     model.get_latest_model_version(client = client, model_name = model)


   
