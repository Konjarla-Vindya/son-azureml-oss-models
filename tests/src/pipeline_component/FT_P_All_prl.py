import os
import json
import concurrent.futures
from fetch_task import HfTask
from box import ConfigBox

test_model_name = os.environ.get('test_model_name')
test_sku_type = os.environ.get('test_sku_type')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_queue = os.environ.get('test_queue')
test_set = os.environ.get('test_set')
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
    model_list = list(queue.models)
    check_mlflow_model = test_model_name
    index = model_list.index(check_mlflow_model)
    print(f"index of {test_model_name} in queue: {index}")

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

# def run_script(script):
#     command = f"python {script}"
#     return_code = os.system(command)
#     return script, return_code

def run_script(script_name):
    subprocess.run(["python", script_name])

def run_fine_tuning_task(task):
    task_script_mapping = {
        "text-classification": "FT_P_TC.py",
        "question-answering": "FT_P_QA.py",
        # Add more mappings as needed
    }

    script_name = task_script_mapping.get(task)
    if script_name:
        run_script(script_name)
    else:
        print(f"No script found for the fine-tune task: {task}")

def run_fine_tuning_tasks(fine_tune_tasks):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_fine_tuning_task, task) for task in fine_tune_tasks]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error running fine-tuning task: {e}")


# def run_fine_tuning(primary_task):
#     task_script_mapping = {
#         "text-classification": ["FT_P_TC.py", "FT_P_QA.py"],
#         "summarization": ["summarization.py", "translation.py"],
#         # Add more tasks and scripts as needed
#     }

#     scripts = task_script_mapping.get(primary_task, [])
#     if scripts:
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = [executor.submit(run_script, script) for script in scripts]

#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     script, return_code = result
#                     print(f"Script '{script}' completed with return code {return_code}")
#                 except Exception as e:
#                     print(f"Error running script '{script}': {e}")
#     else:
#         print(f"No scripts found for the primary task: {primary_task}")

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

    primary_task = HfTask(model_name=test_model_name).get_task()
    print("Task is this: ", primary_task)


    if primary_task:
        # Fetch fine-tune tasks for the specified model
        fine_tune_tasks = foundation_model.properties.get("finetune-recommended-sku", [])
        print("finetune tasks from model card are:", {fine_tune_tasks})

        if fine_tune_tasks:
            # Run fine-tuning tasks in parallel
            run_fine_tuning_tasks(fine_tune_tasks)
        else:
            print(f"No fine-tune tasks found for the model: {model_name}")
    else:
        print(f"No primary task found for the model: {model_name}")

    # if primary_task:
    #     run_fine_tuning(primary_task)
    # else:
    #     print(f"No primary task found for the model: {model_name}")
    
