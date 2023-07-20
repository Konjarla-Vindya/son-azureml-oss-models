
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,AzureCliCredential,
    ClientSecretCredential,
)
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
import json
import os




# model to test    
test_model_name = os.environ.get('test_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# bool to decide if we want to trigger the next model in the queue
# test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue, 
# which means that the first model in the queue is triggered again after the last model is tested
# test_keep_looping = os.environ.get('test_keep_looping')

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group
def get_test_queue():
    config_name = 'test-bert'
    queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)
    
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model   
def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None


# finds the next model in the queue and sends it to github step output 
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
# def set_next_trigger_model(queue):
#     print ("In set_next_trigger_model...")
# # file the index of test_model_name in models list queue dictionary
#     index = queue['models'].index(test_model_name)
#     print (f"index of {test_model_name} in queue: {index}")
# # if index is not the last element in the list, get the next element in the list
#     if index < len(queue['models']) - 1:
#         next_model = queue['models'][index + 1]
#     else:
#         if (test_keep_looping == "true"):
#             next_model = queue[0]
#         else:
#             print ("::warning:: finishing the queue")
#             next_model = ""
# # write the next model to github step output
#     with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
#         print(f'NEXT_MODEL={next_model}')
#         print(f'NEXT_MODEL={next_model}', file=fh)

# # we always test the latest version of the model
def get_latest_model_version(registry_ml_client, model_name):
    print ("In get_latest_model_version...")
    # Getting latest model version from registry is not working, so get all versions and find latest
    model_versions=registry_ml_client.models.list(name=model_name)
    model_version_count=0
    # can't just check len(model_versions) because it is a iterator
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


def create_online_deployment(workspace_ml_client, endpoint, instance_type, latest_model):
    print ("In create_online_deployment...")
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=endpoint.name,
        model=latest_model.id,
        instance_type=instance_type,
        instance_count=1,
    )
    try:
        workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        prase_logs(str(e))
        get_online_endpoint_logs(workspace_ml_client, endpoint.name)
        workspace_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    # online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for demo deployment
    endpoint.traffic = {"demo": 100}
    try:
        workspace_ml_client.begin_create_or_update(endpoint).result()
    except Exception as e:
        print (f"::error:: Could not create deployment\n")
        print (f"{e}\n\n check logs:\n\n")
        get_online_endpoint_logs(workspace_ml_client, endpoint.name)
        workspace_ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        exit (1)
    print(workspace_ml_client.online_deployments.get(name="demo", endpoint_name=endpoint.name))


def delete_online_endpoint(workspace_ml_client, online_endpoint_name):
    try:
        workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
    except Exception as e:
        print (f"::warning:: Could not delete endpoint: : \n{e}")
        exit (0)    



def main():

    # constants
    check_override = True

    # if any of the above are not set, exit with error
    # if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
    #     print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
    #     exit (1)

    queue = get_test_queue()

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False


    # print values of all above variables
    print (f"subscription: {queue['subscription']}")
    # print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    # print (f"test_registry: queue['registry']")
    print (f"registry: {queue['registry']}")
    # print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")


    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    # connect to workspace
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue['subscription'],
        resource_group_name=queue['resource_group'],
        workspace_name=queue['workspace']
    )

    # connect to registry
    registry_ml_client = MLClient(
        credential=credential, 
        registry_name=queue['registry']
    )
    print("reg: ",registry_ml_client)
    print("workspace ", workspace_ml_client)
    # latest_model = get_latest_model_version(registry_ml_client, test_model_name)
    

# endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name

    timestamp = int(time.time())
    online_endpoint_name = "fill-" + str(timestamp)
    print (f"online_endpoint_name: {online_endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
    create_online_endpoint(workspace_ml_client, endpoint)
    create_online_deployment(workspace_ml_client, endpoint, instance_type, latest_model)
#     delete_online_endpoint(workspace_ml_client, online_endpoint_name)
    
        
if __name__ == "__main__":
    main()
