from azureml.core import Experiment, ScriptRunConfig, Workspace, Environment
import os
import json
from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,AzureCliCredential,
    ClientSecretCredential,
)


# workspace = "sonata-test-ws"
# subscription = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
# resource_group = "sonata-test-rg"
# registry = "HuggingFace"
path = "tests/src/register_xlnet_models.py"

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

def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None
    

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



def get_test_queue():
    #config_name = test_queue+'-test'
    #queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}"
    with open(queue_file) as f:
        content = json.load(f)
        return content

def submit_azuremljob():
    pass

if __name__ == "__main__":
    queue = get_test_queue()
    def main():

    # constants
     check_override = True

    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit (1)

    queue = get_test_queue()
    azureml_workspace = Workspace(queue['subscription'], queue['resource_group'], queue['workspace'])
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False

    # if test_trigger_next_model == "true":
    #     set_next_trigger_model(queue)

    # print values of all above variables
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['subscription']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")


  
    # or create a new Pip environment from the requirements.txt file
    env = Environment.from_pip_requirements(name='t5_environment', file_path='requirements/t5_requirements.txt')

    # Register the environment in your workspace
    env.register(workspace=azureml_workspace)
    script_config = ScriptRunConfig(
                            source_directory='.',
                            script='t5_test_translation.py',
                            compute_target='cpu-cluster',
                            environment=env
                            )
    # Create an Experiment
    experiment = Experiment(azureml_workspace, 'my_test_experiment_for_t5_1')
    # Submit the script for execution
    run = experiment.submit(script_config)
    print(run)
    run.wait_for_completion(show_output=True)
    # next_model = set_next_trigger_model(queue)
    # if test_trigger_next_model:
    #     if next_model is not None:
    #         env_file = os.getenv('GITHUB_ENV')
    #         with open(env_file, "a") as myfile:
    #             myfile.write(f"test_model_name={next_model}")
    #             #os.environ["test_model_name"] = 
    # res = os.environ.get("test_model_name")
    # trigger_next_model = os.environ.get("test_trigger_next_model")
    # print(f"Here is the next model to proceed with : {res} and the trigger_next_model value is {trigger_next_model}")
