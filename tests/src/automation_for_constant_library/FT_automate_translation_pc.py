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
