import os
import time
import json
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Input

# Load configuration from a JSON file
with open('configpc.json') as config_file:
    config = json.load(config_file)

subscription_id = config['subscription_id']
resource_group_name = config['resource_group_name']
workspace_name = config['workspace_name']
compute_cluster_size = config['compute_cluster_size']
compute_cluster = config['compute_cluster']
experiment_name = config['experiment_name']

# from azure.ai.ml.entities import MLClient

check_override = True


# def get_error_messages():
#     # load ../config/errors.json into a dictionary
#     with open('../../config/errors.json') as f:
#         return json.load(f)


# error_messages = get_error_messages()

# model to test
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

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group
def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model


def get_sku_override():
    try:
        with open('../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None
    # finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    print("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    index = model_list.index(test_model_name)
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
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


# def create_or_get_compute_target(ml_client,  compute):
#     cpu_compute_target = compute
#     try:
#         compute = ml_client.compute.get(cpu_compute_target)
#     except Exception:
#         print("Creating a new cpu compute target...")
#         compute = AmlCompute(
#             name=cpu_compute_target, size=compute, min_instances=0, max_instances=3, idle_time_before_scale_down = 120
#         )
#         ml_client.compute.begin_create_or_update(compute).result()
#     print(f"New compute target created: {compute.name}")
#     return compute
def run_azure_ml_job(code, command_to_run, environment, compute, environment_variables):
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
    # wait for the job to complete
    workspace_ml_client.jobs.stream(returned_job.name)
    return returned_job.studio_url
def get_latest_model_version(workspace_ml_client, test_model_name):
    print("In get_latest_model_version...")
    version_list = list(workspace_ml_client.models.list(test_model_name))
    
    if len(version_list) == 0:
        print("Model not found in registry")
        foundation_model_name = None  # Set to None if the model is not found
        foundation_model_id = None  # Set id to None as well
    else:
        model_version = version_list[0].version
        foundation_model = workspace_ml_client.models.get(
            test_model_name, model_version)
        print(
            "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                foundation_model.name, foundation_model.version, foundation_model.id
            )
        )
        foundation_model_name = foundation_model.name  # Assign the value to a new variable
        foundation_model_id = foundation_model.id  # Assign the id to a new variable
        # Check if foundation_model_name and foundation_model_id are None or have values
    if foundation_model_name and foundation_model_id:
        print(f"Latest model {foundation_model_name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        print("foundation_model.name:", foundation_model_name)
        print("foundation_model.id:", foundation_model_id)
    else:
        print("No model found in the registry.")
    
    #print(f"Model Config : {latest_model.config}")
    return foundation_model


# Define the Azure Machine Learning client and credential setup
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    
registry_ml_client = MLClient(credential, registry_name="azureml-preview-test1")

# download the dataset using the helper script. This needs datasets library: https://pypi.org/project/datasets/
import os

exit_status = os.system(
    "python ./download_dataset.py --download_dir wmt16-en-ro-dataset"
)
if exit_status != 0:
    raise Exception("Error downloading dataset")
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
# load the train.jsonl, test.jsonl and validation.jsonl files from the ./wmt16-en-ro-dataset/ folder and show first 5 rows
train_df = pd.read_json("./wmt16-en-ro-dataset/train.jsonl", lines=True)
validation_df = pd.read_json("./wmt16-en-ro-dataset/validation.jsonl", lines=True)
test_df = pd.read_json("./wmt16-en-ro-dataset/test.jsonl", lines=True)
# change the frac parameter to control the number of examples to be saved
# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./wmt16-en-ro-dataset folder
frac = 1
train_df.sample(frac=frac).to_json(
    "./wmt16-en-ro-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=frac).to_json(
    "./wmt16-en-ro-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=frac).to_json(
    "./wmt16-en-ro-dataset/small_test.jsonl", orient="records", lines=True
)


# Compute Cluster Creation
try:
    compute = workspace_ml_client.compute.get(compute_cluster)
    print("The compute cluster already exists! Reusing it for the current run")
except Exception as ex:
    print(
        f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
    )
#     # Create a new compute cluster 
#     import ast

# if "computes_allow_list" in foundation_model.tags:
#     computes_allow_list = ast.literal_eval(
#         foundation_model.tags["computes_allow_list"]
#     )  # convert string to python list
#     print(f"Please create a compute from the above list - {computes_allow_list}")
# else:
#     computes_allow_list = None
#     print("Computes allow list is not part of model tags")
# # If you have a specific compute size to work with change it here. By default we use the 1 x V100 compute from the above list
# compute_cluster_size = "Standard_NC6s_v3"

# # If you already have a gpu cluster, mention it here. Else will create a new one with the name 'gpu-cluster-big'
# compute_cluster = "gpu-cluster-big"

# try:
#     compute = workspace_ml_client.compute.get(compute_cluster)
#     print("The compute cluster already exists! Reusing it for the current run")
# except Exception as ex:
#     print(
#         f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
#     )
#     try:
#         print("Attempt #1 - Trying to create a dedicated compute")
#         compute = AmlCompute(
#             name=compute_cluster,
#             size=compute_cluster_size,
#             tier="Dedicated",
#             max_instances=2,  # For multi node training set this to an integer value more than 1
#         )
#         workspace_ml_client.compute.begin_create_or_update(compute).wait()
#     except Exception as e:
        # try:
        #     print(
        #         "Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion."
        #     )
        #     compute = AmlCompute(
        #         name=compute_cluster,
        #         size=compute_cluster_size,
        #         tier="LowPriority",
        #         max_instances=2,  # For multi node training set this to an integer value more than 1
        #     )
        #     workspace_ml_client.compute.begin_create_or_update(compute).wait()
        # except Exception as e:
        #     print(e)
        #     raise ValueError(
        #         f"WARNING! Compute size {compute_cluster_size} not available in workspace"
        #     )


# # Sanity check on the created compute
# compute = workspace_ml_client.compute.get(compute_cluster)
# if compute.provisioning_state.lower() == "failed":
#     raise ValueError(
#         f"Provisioning failed, Compute '{compute_cluster}' is in failed state. "
#         f"please try creating a different compute"
#     )

# if computes_allow_list is not None:
#     computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
#     if compute.size.lower() not in computes_allow_list_lower_case:
#         raise ValueError(
#             f"VM size {compute.size} is not in the allow-listed computes for finetuning"
#         )
# else:
    # # Computes with K80 GPUs are not supported
    # unsupported_gpu_vm_list = [
    #     "standard_nc6",
    #     "standard_nc12",
    #     "standard_nc24",
    #     "standard_nc24r",
    # ]
    # if compute.size.lower() in unsupported_gpu_vm_list:
    #     raise ValueError(
    #         f"VM size {compute.size} is currently not supported for finetuning"
    #     )


# # This is the number of GPUs in a single node of the selected 'vm_size' compute.
# # Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# # Setting this to more than the number of GPUs will result in an error.
# gpu_count_found = False
# workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
# available_sku_sizes = []
# for compute_sku in workspace_compute_sku_list:
#     available_sku_sizes.append(compute_sku.name)
#     if compute_sku.name.lower() == compute.size.lower():
#         gpus_per_node = compute_sku.gpus
#         gpu_count_found = True
# # if gpu_count_found not found, then print an error
# if gpu_count_found:
#     print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
# else:
#     raise ValueError(
#         f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}."
#         f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
#     )
# fetch the pipeline component
pipeline_component_func = registry_ml_client.components.get(
    name="translation_pipeline", label="latest"
)
# fetch Model from WS    
model_name = "t5-small"
foundation_model = workspace_ml_client.models.get(model_name, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
# Input data specification as a dictionary
input_data = {
    "train_file_path": {
        "type": "uri_file",
        "path": "./wmt16-en-ro-dataset/small_train.jsonl"
    },
    "validation_file_path": {
        "type": "uri_file",
        "path": "./wmt16-en-ro-dataset/small_validation.jsonl"
    },
    "test_file_path": {
        "type": "uri_file",
        "path": "./wmt16-en-ro-dataset/small_test.jsonl"
    },
    "evaluation_config": {
        "type": "uri_file",
        "path": "./translation-config.json"
    }
}

# Model Training and Pipeline Setup
@pipeline()
def create_pipeline():
    translation_pipeline = pipeline_component_func(**input_data,
        # specify the foundation model available in the azureml system registry id identified in step #3
        mlflow_model_path=foundation_model.id,
        # huggingface_id = 't5-small', # if you want to use a huggingface model, uncomment this line and comment the above line
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./wmt16-en-ro-dataset/small_train.jsonl"
        ),
        validation_file_path=Input(
            type="uri_file", path="./wmt16-en-ro-dataset/small_validation.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./wmt16-en-ro-dataset/small_test.jsonl"
        ),
        evaluation_config=Input(type="uri_file", path="./translation-config.json"),
        # The following parameters map to the dataset fields
        # source_lang parameter maps to the "en" field in the wmt16 dataset
        source_lang="en",
         # target_lang parameter maps to the "ro" field in the wmt16 dataset
        target_lang="ro",
        # training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of the pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": translation_pipeline.outputs.mlflow_model_folder
    }
    
    # Training parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    metric_for_best_model="bleu",
)
print(f"The following training parameters are enabled - {training_parameters}")

# Optimization parameters - As these parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true"
    )
print(f"The following optimizations are enabled - {optimization_parameters}")

pipeline_object = create_pipeline()

# Don't use cached results from previous jobs
pipeline_object.settings.force_rerun = True

# Set continue on step failure to False
pipeline_object.settings.continue_on_step_failure = False

# Submit the pipeline job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)

# Wait for the pipeline job to complete
workspace_ml_client.jobs.stream(pipeline_job.name)
