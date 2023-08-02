print("PY Started")
from azure.ai.ml import MLClient
from azureml.core import Workspace
import mlflow
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

def get_test_queue():
    config_name = test_queue+'-test'
    queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)

def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

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


def get_instance_type(latest_model, sku_override, registry_ml_client, check_override):
    # determine the instance_type from the sku templates available in the model properties
    # 1. get the template name matching the sku_type
    # 2. look up template-sku.json to find the instance_type
    model_properties = str(latest_model.properties)
    # escape double quotes in model_properties
    model_properties = model_properties.replace('"', '\\"')
    # replace single quotes with double quotes in model_properties
    model_properties = model_properties.replace("'", '"')
    # convert model_properties to dictionary
    model_properties_dict=json.loads(model_properties)
    sku_templates = model_properties_dict['skuBasedEngineIds']
    # split sku_templates by comma into a list
    sku_templates_list = sku_templates.split(",")
    # find the sku_template that has sku_type as a substring
    sku_template = next((s for s in sku_templates_list if test_sku_type in s), None)
    if sku_template is None:
        print (f"::error:: Could not find sku_template for {test_sku_type}")
        exit (1)
    print (f"sku_template: {sku_template}")
    # split sku_template by / and get the 5th element into a variable called template_name
    template_name = sku_template.split("/")[5]
    print (f"template_name: {template_name}")
    template_latest_version=get_latest_model_version(registry_ml_client, template_name)

    #print (template_latest_version.properties) 
    # split the properties by by the pattern "DefaultInstanceType": " and get 2nd element
    # then again split by " and get the first element
    instance_type = str(template_latest_version.properties).split('"DefaultInstanceType": "')[1].split('"')[0]
    print (f"instance_type: {instance_type}")

    if instance_type is None:
        print (f"::error:: Could not find instance_type for {test_sku_type}")
        exit (1)

    if check_override:
        if latest_model.name in sku_override:
            instance_type = sku_override[test_model_name]['sku']
            print (f"overriding instance_type: {instance_type}")
    
    return instance_type

    
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
        instance_type=instance_type,
        instance_count=1,
    )
    workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    endpoint.traffic = {"demo": 100}
    workspace_ml_client.begin_create_or_update(endpoint).result()
   


def sample_inference(latest_model,registry, workspace_ml_client, online_endpoint_name):
    tags = str(latest_model.tags)
    tags = tags.replace("'", '"')
    tags_dict=json.loads(tags)
    task = tags_dict['task']
    print (f"task: {task}")
    scoring_file = f"../../config/sample_inputs/{registry}/{task}.json"
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

# def delete_online_endpoint(workspace_ml_client, online_endpoint_name):
#     try:
#         workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
#     except Exception as e:
#         print (f"::warning:: Could not delete endpoint: : \n{e}")
#         exit (0)    


def main():
    
    model = Model(model_name=test_model_name)
    print (model)
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['subscription']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue['subscription'],
        resource_group_name=queue['resource_group'],
        workspace_name=queue['workspace']
    )
    registry_ml_client = MLClient(
        credential=credential, 
        registry_name=registry
    )
   

    latest_model = get_latest_model_version(registry_ml_client, test_model_name)
    instance_type = get_instance_type(latest_model, sku_override, registry_ml_client, check_override)
    print("the task is:",task)
    timestamp = int(time.time())
    online_endpoint_name = task + str(timestamp)
    print (f"online_endpoint_name: {online_endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
   
    print("latest_model:",latest_model)
    print("endpoint name:",endpoint)
    sample_inference(latest_model,registry, workspace_ml_client, online_endpoint_name)


if __name__ == "__main__":
    main()
    
