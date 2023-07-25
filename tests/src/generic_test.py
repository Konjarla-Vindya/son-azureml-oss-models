from azureml.core import Experiment, ScriptRunConfig, Workspace
import os
import json

workspace = "sonata-test-ws"
subscription = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
resource_group = "sonata-test-rg"
registry = "HuggingFace"
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



def get_test_queue():
    #config_name = test_queue+'-test'
    #queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}"
    with open(queue_file) as f:
        content = json.load(f)
        return content

def submit_azuremljob():
    pass
# def set_next_trigger_model(queue):
#     print ("In set_next_trigger_model...")
#     # file the index of test_model_name in models list queue dictionary
#     index = queue['models'].index(test_model_name)
#     print (f"index of {test_model_name} in queue: {index}")
# # if index is not the last element in the list, get the next element in the list
#     if index < len(queue['models']) - 1:
#         next_model = queue['models'][index + 1]
#     else:
#         next_model = ""
#         # if (test_keep_looping == "true"):
#         #     next_model = queue[0]
#         # else:
#         #     print ("::warning:: finishing the queue")
#         #     next_model = ""
#         test_trigger_next_model = False
#         #os.environ["test_trigger_next_model"] = False
#         env_file = os.getenv('GITHUB_ENV')
#         with open(env_file, "a") as myfile:
#             myfile.write(f"test_trigger_next_model={test_trigger_next_model}")
#     with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
#         print(f'NEXT_MODEL={next_model}')
#         print(f'NEXT_MODEL={next_model}', file=fh)
#     return next_model

if __name__ == "__main__":
    queue = get_test_queue()
    azureml_workspace = Workspace(subscription, resource_group, workspace)
    script_config = ScriptRunConfig(
                            source_directory='./tests/src/',
                            script='register_xlnet_models.py',
                            compute_target='cpu-cluster',
                            
                            )
    # Create an Experiment
    experiment = Experiment(azureml_workspace, 'my_test_experiment_for_run')
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
    