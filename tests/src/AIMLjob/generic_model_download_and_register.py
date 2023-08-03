from azure.ai.ml import MLClient
from transformers import AutoModel,AutoTokenizer
#import transformers
from azureml.core import Workspace
#from azureml.core import Workspace
#from azureml.mlflow import get_mlflow_tracking_uri
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
import mlflow
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
import os, json
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)

test_model_name = os.environ.get('test_model_name')
subscription = os.environ.get('subscription')
resource_group = os.environ.get('resource_group')
workspace_name = os.environ.get('workspace')
registry=os.environ.get('registry')

class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
    
    def download_model_and_tokenizer(self)->dict:
        model = AutoModel.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_and_tokenizer = {"model":model, "tokenizer":tokenizer}
        return model_and_tokenizer
    
    def register_model_in_workspace(self, model_and_tokenizer):
        #task = self.queue.models[self.model_name].task
        artifact_path = self.model_name + "-artifact"
        registered_model_name = self.model_name
        # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        mlflow.transformers.log_model(
            transformers_model = model_and_tokenizer,
            #task=task,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        mlflow.transformers.save_model(model_and_tokenizer,path=artifact_path)
        registered_model = mlflow.transformers.load_model(artifact_path)
        
        shutil.rmtree(artifact_path)
        return registered_model
    
    def download_and_register_model(self)->dict :
        model_and_tokenizer = self.download_model_and_tokenizer()
        # workspace = Workspace(
        #         subscription_id = subscription,
        #         resource_group = resource_group,
        #         workspace_name = workspace_name
        #     )
        self.register_model_in_workspace(model_and_tokenizer)
        return model_and_tokenizer

def get_latest_model_version(registry_ml_client, model_name):
    print ("In get_latest_model_version...")
    model_versions=registry_ml_client.models.list(name=model_name)
    model_version_count=0
    models = []
    for model in model_versions:
        model_version_count = model_version_count + 1
        models.append(model)
    sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
    latest_model = sorted_models[0]
    print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
    print(latest_model)
    return latest_model
def create_online_endpoint(workspace_ml_client, endpoint):
    print ("In create_online_endpoint...")
    try:
        workspace_ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    except Exception as e:
        print (f"::error:: Could not create endpoint: \n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        exit (1)

    print(workspace_ml_client.online_endpoints.get(name=endpoint.name))


def create_online_deployment(workspace_ml_client, endpoint, latest_model):
    print ("In create_online_deployment...")
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=endpoint.name,
        model=latest_model.id,
        instance_type="Standard_DS4_v2",
        instance_count=1,
    )
    workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    endpoint.traffic = {"demo": 100}
    workspace_ml_client.begin_create_or_update(endpoint).result()
   

def inference(self,download_model_and_tokenizer):

        if self.task_name in ("text-generation","translation","summarization"):   
            input_ids = download_model_and_tokenizer["tokenizer"](self.input_text, return_tensors="pt").input_ids 
            ids = download_model_and_tokenizer["model"].generate(input_ids)
            prediction  = download_model_and_tokenizer["tokenizer"].decode(ids[0], skip_special_tokens=True)

        elif self.task_name=="text-classification":
            inputs = download_model_and_tokenizer["tokenizer"](self.input_text, return_tensors="pt")
            logits = download_model_and_tokenizer["model"](**inputs).logits
            predicted_class_id = logits.argmax(axis=-1).item()
            prediction=download_model_and_tokenizer["model"].config.id2label[predicted_class_id]
        
        elif self.task_name=="fill-mask":
            inputs = download_model_and_tokenizer["tokenizer"](self.input_text, return_tensors="pt") 
            logits = download_model_and_tokenizer["model"](**inputs).logits
            mask_token_index = torch.where(inputs["input_ids"] == download_model_and_tokenizer["tokenizer"].mask_token_id)[1]
            mask_token_logits = logits[0, mask_token_index, :]
            top_1_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
            prediction  = self.input_text.replace(download_model_and_tokenizer["tokenizer"].mask_token, download_model_and_tokenizer["tokenizer"].decode([top_1_tokens[0]]))

        elif self.task_name=="question-answering":
            inputs = download_model_and_tokenizer["tokenizer"](self.input_text,self.context, return_tensors="pt") 
            outputs = download_model_and_tokenizer["model"](**inputs)
            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            prediction = download_model_and_tokenizer["tokenizer"].decode(predict_answer_tokens)
        
        elif self.task_name == "token-classification":
            inputs = download_model_and_tokenizer["tokenizer"](self.input_text, return_tensors="pt")
            logits = download_model_and_tokenizer["model"](**inputs).logits
            predicted_ids = torch.argmax(logits, dim=2)
            prediction=[download_model_and_tokenizer["model"].config.id2label[t.item()] for t in predicted_ids[0]]

        else:
            prediction = None

        return prediction
    
def local_inference(latest_model,registry, workspace_ml_client, online_endpoint_name):
    # get the task tag from the latest_model.tags
    tags = str(latest_model.tags)
    # replace single quotes with double quotes in tags
    tags = tags.replace("'", '"')
    # convert tags to dictionary
    tags_dict=json.loads(tags)
    task = tags_dict['task']
    print (f"task: {task}")
    scoring_file = f"../../config/sample_inputs/{registry}/{task}.json"
    # check of scoring_file exists
    try:
        with open(scoring_file) as f:
            scoring_input = json.load(f)
            print (f"scoring_input file:\n\n {scoring_input}\n\n")
    except Exception as e:
        print (f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")

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
        print (f"::error:: Could not invoke endpoint: \n")
        print (f"{e}\n\n check logs:\n\n")
        # get_online_endpoint_logs(workspace_ml_client, online_endpoint_name)
if __name__ == "__main__":
    model = Model(model_name=test_model_name)
    # try:
    #     credential = DefaultAzureCredential()
    #     credential.get_token("https://management.azure.com/.default")
    # except Exception as ex:
    #     print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
    #     exit (1)
    # workspace_ml_client = MLClient(
    #     credential=credential, 
    #     subscription_id=queue['subscription'],
    #     resource_group_name=queue['resource_group'],
    #     workspace_name=queue['workspace']
    # )
    # registry_ml_client = MLClient(
    #     credential=credential, 
    #     registry_name=registry
    # )
    download_model_and_tokenizer=model.download_and_register_model()

    # latest_model = get_latest_model_version(registry_ml_client, test_model_name)
    # #download_and_register_model()
    #  # get the task tag from the latest_model.tags
    # tags = str(latest_model.tags)
    # # replace single quotes with double quotes in tags
    # tags = tags.replace("'", '"')
    # # convert tags to dictionary
    # tags_dict=json.loads(tags)
    # task = tags_dict['task']
    # print("the task is:",task)
    # # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
    # timestamp = int(time.time())
    # online_endpoint_name = task + str(timestamp)
    # print (f"online_endpoint_name: {online_endpoint_name}")
    # endpoint = ManagedOnlineEndpoint(
    #     name=online_endpoint_name,
    #     auth_mode="key",
    # )
   
    # print("latest_model:",latest_model)
    # print("endpoint name:",endpoint)
    prediction = inference(Model_Tokenziner)
    print(prediction)
    # create_online_endpoint(workspace_ml_client, endpoint)
    # create_online_deployment(workspace_ml_client, endpoint, latest_model)
    # local_inference(latest_model,registry, workspace_ml_client, online_endpoint_name)
    
