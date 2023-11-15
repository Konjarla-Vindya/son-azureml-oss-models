from fetch_task import HfTask
import os
from box import ConfigBox
import json


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

def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))


def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None

def set_next_trigger_model(queue):
    print("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    # check_mlflow_model = "MLFlow-Batch-"+test_model_name
    check_mlflow_model = test_model_name
    index = model_list.index(check_mlflow_model)
    #index = model_list.index(test_model_name)
    #index = model_list.index(test_model_name)
    print(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print("::warning:: finishing the queue")
            next_model = ""

    
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)
        

def run_fine_tuning(primary_task):
    print("entered function")
    # Import necessary modules
    import subprocess
    print("picking task")

    # Mapping of primary tasks to corresponding script names
    task_script_mapping = {
        "text-classification": ["FT_P_TC.py", "FT_P_QA.py"],
        "summarization": ["summarization.py", "translation.py"],
        "translation": ["translation.py", "summarization.py"],
        "fill-mask": ["text-classification.py", "question-answering.py", "token-classification"],
        "question-answering": ["text-classification.py", "question-answering.py", "token-classification.py"],
        "text-generation": ["text-classification.py", "token-classification.py"],
        "token-classification": ["token-classification.py", "text-classification.py"]
    }

    # Get the script names based on the primary task
    scripts = task_script_mapping.get(primary_task, [])

    # If scripts exist, run each of them
    if scripts:
        for script_name in scripts:
            subprocess.run(["python", script_name])
    else:
        print(f"No scripts found for the primary task: {primary_task}")

if __name__ == "__main__":

    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        exit(1)

    queue = get_test_queue()

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)

    print(f"test_subscription_id: {queue['subscription']}")
    print(f"test_resource_group: {queue['resource_group']}")
    print(f"test_workspace_name: {queue['workspace']}")
    print(f"test_model_name: {test_model_name}")
    print(f"test_sku_type: {test_sku_type}")
    print(f"test_registry: {queue['registry']}")
    print(f"test_trigger_next_model: {test_trigger_next_model}")
    print(f"test_queue: {test_queue}")
    print(f"test_set: {test_set}")
    print("Here is my test model name: ", test_model_name)
    # # Replace 'model1' with the actual model name or fetch it dynamically
    # model_name = "model1"

    primary_task = HfTask(model_name=test_model_name).get_task()
    print("Task is this: ", primary_task)

    # # Get the primary task for the specified model
    # primary_task = model_primary_tasks.get(model_name)

    if primary_task:
        # Run fine-tuning based on the primary task
        run_fine_tuning(primary_task)
    else:
        print(f"No primary task found for the model: {model_name}")
