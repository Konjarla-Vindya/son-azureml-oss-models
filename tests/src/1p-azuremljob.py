from azureml.core import Experiment, ScriptRunConfig, Workspace, Environment
import os
import json
# from azure.identity import (
#     DefaultAzureCredential,
#     ClientSecretCredential
# )
from azure.identity import DefaultAzureCredential,AzureCliCredential 

workspace = "sonata-test-ws"
subscription = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
resource_group = "sonata-test-rg"
registry = "HuggingFace"

test_model_name = os.environ.get('test_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# bool to decide if we want to trigger the next model in the queue
test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue, 
# which means that the first model in the queue is triggered again after the last model is tested
test_keep_looping = os.environ.get('test_keep_looping')

def get_test_queue():
    queue_file = f"../config/queue/{test_set}/{test_queue}"
    with open(queue_file) as f:
        content = json.load(f)
        return content

if __name__ == "__main__":
    queue = get_test_queue()
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)
        
    azureml_workspace = Workspace(subscription, resource_group, workspace)
    # or create a new Pip environment from the requirements.txt file
    myenv = Environment.get(workspace=azureml_workspace, name="bert_environment")
    env = myenv.from_pip_requirements(name="bert_environment", file_path='requirements/bert_requirements.txt')

    # Register the environment in your workspace
    env.register(workspace=azureml_workspace)
    script_config = ScriptRunConfig(
                            source_directory='./AML_Jobs',
                            script='BertJob.py',
                            compute_target='cpu-cluster',
                            environment=env
                            )
    # Create an Experiment
    experiment = Experiment(azureml_workspace, 'my_experiment_for_bertp')
    # Submit the script for execution
    run = experiment.submit(script_config)
    print(run)
    run.wait_for_completion(show_output=True)   
