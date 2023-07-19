from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,AzureCliCredential 
from azureml.core import Workspace
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from azureml.mlflow import get_mlflow_tracking_uri
import mlflow

checkpoint = "bert-base-uncased"
registered_model_name = "bert_registered"

def set_tracking_uri(credential):
    subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
    resource_group = 'sonata-test-rg'
    workspace_name = 'sonata-test-ws'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    workspace_ml_client = MLClient(
                        credential, subscription_id, resource_group, ws
                    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    
def download_and_register_model():
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    mlflow.transformers.log_model(
            transformers_model = {"model" : model, "tokenizer":tokenizer},
            task="fill-mask",
            artifact_path="Bert_artifact",
            registered_model_name=registered_model_name
    )
    
def get_latest_version_model(registry_ml_client):
    model_versions = list(registry_ml_client.models.list(registered_model_name))
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
    credential = DefaultAzureCredential()
    set_tracking_uri(credential)
    download_and_register_model()
    
    # connect to registry
    registry_ml_client = MLClient(
        credential=credential, 
        registry_name="sonata-test-reg"
    )
    latest_model = get_latest_version_model(registry_ml_client)
    


    # def main():

    # # constants
    # check_override = True

    # # if any of the above are not set, exit with error
    # if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
    #     print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
    #     exit (1)

    # queue = get_test_queue()

    # sku_override = get_sku_override()
    # if sku_override is None:
    #     check_override = False

    # if test_trigger_next_model == "true":
    #     set_next_trigger_model(queue)

    # # print values of all above variables
    # print (f"test_subscription_id: {queue['subscription']}")
    # print (f"test_resource_group: {queue['subscription']}")
    # print (f"test_workspace_name: {queue['workspace']}")
    # print (f"test_model_name: {test_model_name}")
    # print (f"test_sku_type: {test_sku_type}")
    # print (f"test_registry: queue['registry']")
    # print (f"test_trigger_next_model: {test_trigger_next_model}")
    # print (f"test_queue: {test_queue}")
    # print (f"test_set: {test_set}")
    
    # try:
    #     credential = AzureCliCredential()
    #     credential.get_token("https://management.azure.com/.default")
    # except Exception as ex:
    #     print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
    #     exit (1)

    # # connect to workspace
    # workspace_ml_client = MLClient(
    #     credential=credential, 
    #     subscription_id=queue['subscription'],
    #     resource_group_name=queue['resource_group'],
    #     workspace_name=queue['workspace']
    # )

    # # connect to registry
    # registry_ml_client = MLClient(
    #     credential=credential, 
    #     registry_name="sonata-test-reg"
    # )

   
