from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import XLNetForSequenceClassification,XLNetTokenizer
from azureml.mlflow import get_mlflow_tracking_uri
import mlflow
# Replace with your Azure ML workspace details
# subscription_id = "bb9cf94f-f06a-49eb-a8e9-e63654d7257b"
# resource_group = "Free"
# workspace_name = "Trial"
# credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
checkpoint = "xlnet-base-cased"
registered_model_name = "Xlnet_registered"

def set_tracking_uri(credential):
    subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
    resource_group = 'sonata-test-rg'
    workspace_name = 'sonata-test-ws'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    workspace_ml_client = MLClient(
                        credential, subscription_id, resource_group, ws
                    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    #print("Reaching here in the set tracking uri method")


def download_and_register_model():
    model = XLNetForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = XLNetTokenizer.from_pretrained(checkpoint)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="text-classification",
            artifact_path="XlNetClassification_artifact",
            registered_model_name=registered_model_name
    )
    #print("Reaching here in the download and register model methos")
    
def get_latest_version_model(registry_ml_client):
    model_versions = list(registry_ml_client.models.list(registered_model_name))
    print(f"Here are the registered model versions : {model_versions}")
    if len(model_versions) == 0:
        print("There is no previously registered model")
    else:
        models = []
        for model in model_versions:
            model_version_count = model_version_count + 1
            models.append(model)
        # Sort models by creation time and find the latest model
        sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
        latest_model = sorted_models[0]
        print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
        print(latest_model)
        return latest_model
    return None

if __name__ == "__main__":
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    #credential = AzureCliCredential()
    except Exception as e:
        print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")
    set_tracking_uri(credential)
    download_and_register_model()
    # connect to registry
    registry_ml_client = MLClient(
        credential=credential, 
        registry_name="sonata-test-reg"
    )
    latest_model = get_latest_version_model(registry_ml_client)
