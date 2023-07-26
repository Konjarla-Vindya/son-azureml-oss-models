#import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
import mlflow

#Enter details of your Azure Machine Learning workspace
subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
resource_group = 'sonata-test-rg'
workspace = 'sonata-test-ws'
#ws = Workspace.from_config()

def connect_to_workspace():
    #connect to the workspace
    #ml_client = MLClient(DefaultAzureCredential(), Workspace.from_config())

    ws = Workspace(subscription_id, resource_group, workspace)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

def specify_compute(ml_client):
    # specify aml compute name.
    cpu_compute_target = "cpu-cluster"

    try:
        ml_client.compute.get(cpu_compute_target)
    except Exception:
        print("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()

def define_command():
    # define the command
    command_job = command(
        code="./",
        command="python Bert.py",
        #--cnn_dailymail ${{inputs.cnn_dailymail}}",
        environment="gpt2-venv:6", #"EnvTest:1",
        # inputs={
        #     "cnn_dailymail": Input(
        #         type="uri_file",
        #         path="https://datasets-server.huggingface.co/rows?dataset=cnn_dailymail&config=3.0.0&split=validation&offset=0&limit=5",
        #     )
        # },
        compute="cpu-cluster",
    )
    return command_job

if __name__ == "__main__":
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    #credential = AzureCliCredential()
    except Exception as e:
        print (f"::warning:: Getting Exception in the default azure credential and here is the exception log : \n{e}")
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
    connect_to_workspace()
    specify_compute(ml_client)
    command_job = define_command()
    # submit the command
    returned_job = ml_client.jobs.create_or_update(command_job)
    # get a URL for the status of the job
    returned_job.studio_url