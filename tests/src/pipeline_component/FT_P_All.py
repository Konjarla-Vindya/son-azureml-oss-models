from fetch_tasks import HfTask
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

def run_fine_tuning(primary_task):
    # Import necessary modules
    import subprocess

    # Mapping of primary tasks to corresponding script names
    task_script_mapping = {
        "text_classification": ["text_classification.py", "question_answering.py", "token_classification.py"],
        "summarization": ["summarization.py", "translation.py"],
        "translation": ["translation.py", "summarization.py"],
        # ... add more tasks and script names as needed
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
