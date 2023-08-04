from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
import mlflow
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
import os, json
import time, sys
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
import transformers
import datetime
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    ModelConfiguration,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer
)
def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../../config/errors.json') as f:
        return json.load(f)
    
error_messages = get_error_messages()
test_model_name = os.environ.get('test_model_name')
test_sku_type = os.environ.get('test_sku_type')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_queue = os.environ.get('test_queue')
test_set = os.environ.get('test_set')
test_keep_looping = os.environ.get('test_keep_looping')

def get_test_queue():
    config_name = test_queue+'-test'
    #queue_file1 = f"../../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)
    

def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

def set_next_trigger_model(queue):
    print ("In set_next_trigger_model...")
    index = queue['models'].index(test_model_name)
    print (f"index of {test_model_name} in queue: {index}")
    if index < len(queue['models']) - 1:
        next_model = queue['models'][index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print ("::warning:: finishing the queue")
            next_model = ""
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def get_latest_model_version(workspace_ml_client, model_name):
    print ("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(model_name))
    if len(version_list) == 0:
        print("Model not found in registry")
    else:
        model_version = version_list[0].version
    latest_model = workspace_ml_client.models.get(model_name, model_version)
    print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
        latest_model.name, latest_model.version, latest_model.id
        )
        )
    return latest_model

    if check_override:
        if latest_model.name in sku_override:
            instance_type = sku_override[test_model_name]['sku']
            print (f"overriding instance_type: {instance_type}")


def create_or_get_compute_target(ml_client):
    cpu_compute_target = "cpu-cluster"
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        print("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size="Standard_DS4_v2", min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()
    
    return compute


def run_azure_ml_job(code, command_to_run, environment, compute,environment_variables):
    command_job = command(
        code=code,
        command=command_to_run,
        environment=environment,
        compute=compute,
        environment_variables=environment_variables
    )
    return command_job

def create_and_get_job_studio_url(command_job, workspace_ml_client):
   
    #ml_client = mlflow.tracking.MlflowClient()
    returned_job = workspace_ml_client.jobs.create_or_update(command_job)
    workspace_ml_client.jobs.stream(returned_job.name)
    return returned_job.studio_url
# studio_url = create_and_get_job_studio_url(command_job)
# print("Studio URL for the job:", studio_url)

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
    model_for_package = Model(name=latest_model.name, version=latest_model.version, type=AssetTypes.MLFLOW_MODEL)
    model_configuration = ModelConfiguration(mode="download")
    package_name = f"package-v2-{latest_model.name}"
    package_config = ModelPackage(
                        target_environment_name=package_name,
                        inferencing_server=AzureMLOnlineInferencingServer(),
                        model_configuration=model_configuration
            )

    model_package = workspace_ml_client.models.package(
                            latest_model.name,
                            latest_model.version,
                            package_config
                        )

        

    workspace_ml_client.begin_create_or_update(endpoint).result()
    deployment_name = latest_model.name

    deployment_config = ManagedOnlineDeployment(
                name = deployment_name,
                model=latest_model.id,
                endpoint_name=online_endpoint_name,
                environment=model_package,
                instance_count=1
            )

    deployment = workspace_ml_client.online_deployments.begin_create_or_update(deployment_config).result()




    # demo_deployment = ManagedOnlineDeployment(
    #     name="demo",
    #     endpoint_name=endpoint.name,
    #     model=latest_model.id,
    #     instance_type="Standard_DS4_v2",
    #     instance_count=1,
    # )
    # workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).result()
    # # endpoint.traffic = {"demo": 100}
    # # workspace_ml_client.begin_create_or_update(endpoint).result()
   


def sample_inference(latest_model,registry, workspace_ml_client, online_endpoint_name):
    # get the task tag from the latest_model.tags
    # tags = str(latest_model.tags)
    # # replace single quotes with double quotes in tags
    # tags = tags.replace("'", '"')
    # # convert tags to dictionary
    # tags_dict=json.loads(tags)
    # task = tags_dict['task']
    # print (f"task: {task}")
    task=latest_model.flavors["transformers"]["task"]
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
    
def main():
    
    check_override = True

    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit (1)

    queue = get_test_queue()

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    print("Present test model name : ",test_model_name)

    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_sstring: {queue['teststring']}")
    print (f"test_int: {queue['testint']}")
    print (f"test_spcl {queue['testspcl']}")
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
    ws = Workspace(subscription_id=queue['subscription'],
        resource_group=queue['resource_group'],
        workspace_name=queue['workspace'])

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    registry_ml_client = MLClient(
        credential=credential, 
        registry_name=queue['registry']
    )

    
    latest_model = get_latest_model_version(workspace_ml_client, test_model_name)
    #download_and_register_model()
    #  # get the task tag from the latest_model.tags
    # tags = str(latest_model.tags)
    # # replace single quotes with double quotes in tags
    # tags = tags.replace("'", '"')
    # # convert tags to dictionary
    # tags_dict=json.loads(tags)
    # task = tags_dict['task']
    # print("the task is:",task)

    
    task=latest_model.flavors["transformers"]["task"]
    # compute_target = create_or_get_compute_target(workspace_ml_client)
    # environment_variables = {"test_model_name": test_model_name, 
    #        "subscription": queue['subscription'],
    #        "resource_group": queue['resource_group'],
    #        "workspace": queue['workspace']}
    # command_job = run_azure_ml_job(code="./", command_to_run="python generic_model_download_and_register.py", environment="env:5", compute="cpu-cluster",environment_variables=environment_variables)
    # create_and_get_job_studio_url(command_job, workspace_ml_client)
    
#     # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
# # generic_model_download_and_register
#     # automated_distlbert
    timestamp = int(time.time())
    online_endpoint_name =task + str(timestamp)
    print (f"online_endpoint_name: {online_endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
   
    print("latest_model---------------------------------------------:",latest_model,"-----------------------------")
    print(latest_model.flavors["transformers"]["task"])
    
    # print("endpoint name:",endpoint)
    create_online_endpoint(workspace_ml_client, endpoint)
    create_online_deployment(workspace_ml_client, endpoint, latest_model)
    # sample_inference(latest_model,queue['registry'], workspace_ml_client, online_endpoint_name)

if __name__ == "__main__":
    main()
    
