from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import XLNetForSequenceClassification,XLNetTokenizer
import mlflow
# Replace with your Azure ML workspace details
# subscription_id = "bb9cf94f-f06a-49eb-a8e9-e63654d7257b"
# resource_group = "Free"
# workspace_name = "Trial"
# credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
def set_tracking_uri():
    subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
    resource_group = 'sonata-test-rg'
    workspace_name = 'sonata-test-ws'

    credential = AzureCliCredential()
    ws = Workspace(subscription_id, resource_group, workspace_name)
    workspace_ml_client = MLClient(
                    credential, subscription_id, resource_group, ws
                )

    #mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    #print("Reaching here in the set tracking uri method")

def download_and_register_model():
    checkpoint = "xlnet-base-cased"
    model = XLNetForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = XLNetTokenizer.from_pretrained(checkpoint)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="Classification",
            artifact_path="XlNetClassification_artifact",
            registered_model_name="Xlnet_registered"
    )
    #print("Reaching here in the download and register model methos")

if __name__ == "__main__":
    set_tracking_uri()
    download_and_register_model()
