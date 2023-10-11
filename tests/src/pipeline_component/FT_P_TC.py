#from model_inference_and_deployment import ModelInferenceAndDeployemnt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from azure.ai.ml import command
import mlflow
import json
import os
import sys
from box import ConfigBox
from mlflow.tracking.client import MlflowClient
from azureml.core import Workspace, Environment
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
from azure.ai.ml.entities import AmlCompute
import time
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
import ast
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


# def load_model(model_detail):
#     loaded_model = mlflow.transformers.load_model(model_uri=model_detail.source, return_type="pipeline")
#     print("Inside load model")
#     print("loaded_model---------------",loaded_model)
#     return loaded_model
# def classify_text(texts, FT_loaded_model, fine_tuned_tokenizer):    
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
#     with torch.no_grad():
#         logits = model(**inputs).logits  
#     predicted_labels = torch.argmax(logits, dim=1).tolist()
#     return predicted_labels    



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


def get_training_and_optimization_parameters(foundation_model):
    # Training parameters
    training_parameters = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 2e-5,
        "metric_for_best_model": "f1_macro",
    }
    print(f"The following training parameters are enabled - {training_parameters}")

    # Optimization parameters
    if "model_specific_defaults" in foundation_model.tags:
        optimization_parameters = ast.literal_eval(foundation_model.tags["model_specific_defaults"])
    else:
        optimization_parameters = {
            "apply_lora": "true",
            "apply_deepspeed": "true",
            "apply_ort": "true",
        }
    print(f"The following optimizations are enabled - {optimization_parameters}")

    return training_parameters, optimization_parameters


# def find_gpus_in_compute(workspace_ml_client, compute):
#     gpu_count_found = False
#     workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
#     available_sku_sizes = []
#     gpus_per_node = 0

#     for compute_sku in workspace_compute_sku_list:
#         available_sku_sizes.append(compute_sku.name)
#         if compute_sku.name.lower() == compute.size.lower():
#             gpus_per_node = compute_sku.gpus
#             gpu_count_found = True

#     if gpu_count_found:
#         print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
#         return gpus_per_node
#     else:
#         raise ValueError(
#             f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}. "
#             f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
#         )



def create_or_get_aml_compute(workspace_ml_client, compute_cluster, compute_cluster_size, computes_allow_list=None):
    try:
        compute = workspace_ml_client.compute.get(compute_cluster)
        print(f"The compute cluster '{compute_cluster}' already exists! Reusing it for the current run")
    except Exception as ex:
        print(f"Looks like the compute cluster '{compute_cluster}' doesn't exist. Creating a new one with compute size '{compute_cluster_size}'!")

        # Define a list of VM sizes that are not supported for finetuning
        unsupported_gpu_vm_list = ["standard_nc6", "standard_nc12", "standard_nc24", "standard_nc24r"]

        try:
            print("Attempt #1 - Trying to create a dedicated compute")
            tier = "Dedicated"
            if compute_cluster_size.lower() in unsupported_gpu_vm_list:
                raise ValueError(f"VM size '{compute_cluster_size}' is not supported for finetuning.")
        except ValueError as e:
            print(e)
            raise

        try:
            print("Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion.")
            tier = "LowPriority"
            if compute_cluster_size.lower() in unsupported_gpu_vm_list:
                raise ValueError(f"VM size '{compute_cluster_size}' is not supported for finetuning.")
        except ValueError as e:
            print(e)
            raise

        # Provision the compute
        compute = AmlCompute(
            name=compute_cluster,
            size=compute_cluster_size,
            tier=tier,
            max_instances=2,  # For multi-node training, set this to an integer value more than 1
        )
        workspace_ml_client.compute.begin_create_or_update(compute).wait()

    # Sanity check on the created compute
    compute = workspace_ml_client.compute.get(compute_cluster)

    if compute.provisioning_state.lower() == "failed":
        raise ValueError(f"Provisioning failed. Compute '{compute_cluster}' is in a failed state. Please try creating a different compute.")

    if computes_allow_list is not None:
        computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
        if compute.size.lower() not in computes_allow_list_lower_case:
            raise ValueError(f"VM size '{compute.size}' is not in the allow-listed computes for finetuning.")
    
    # Determine the number of GPUs in a single node of the selected 'compute_cluster_size' compute
    gpu_count_found = False
    workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
    available_sku_sizes = []

    for compute_sku in workspace_compute_sku_list:
        available_sku_sizes.append(compute_sku.name)
        if compute_sku.name.lower() == compute.size.lower():
            gpus_per_node = compute_sku.gpus
            gpu_count_found = True

    # If the GPU count is not found, print an error
    if gpu_count_found:
        print(f"Number of GPUs in compute '{compute_cluster}': {gpus_per_node}")
    else:
        raise ValueError(f"Number of GPUs in compute '{compute_cluster}' not found. Available skus are: {available_sku_sizes}. This should not happen. Please check the selected compute cluster: {compute_cluster} and try again.")
    
    return compute, gpus_per_node, compute_cluster




def create_and_run_azure_ml_pipeline(
    foundation_model,
    compute_cluster,
    gpus_per_node,
    training_parameters,
    optimization_parameters,
    experiment_name,
):
    # Fetch the pipeline component
    pipeline_component_func = registry_ml_client.components.get(
        name="text_classification_pipeline", label="latest"
    )

    # Define the pipeline job
    @pipeline()
    def create_pipeline():
        text_classification_pipeline = pipeline_component_func(
            mlflow_model_path=foundation_model.id,
            compute_model_import=compute_cluster,
            compute_preprocess=compute_cluster,
            compute_finetune=compute_cluster,
            compute_model_evaluation=compute_cluster,
            train_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_train.jsonl"
            ),
            validation_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_validation.jsonl"
            ),
            test_file_path=Input(
                type="uri_file", path="./emotion-dataset/small_test.jsonl"
            ),
            evaluation_config=Input(
                type="uri_file", path="./emotion-dataset/text-classification-config.json"
            ),
            sentence1_key="text",
            label_key="label_string",
            number_of_gpu_to_use_finetuning=gpus_per_node,
            **training_parameters,
            **optimization_parameters
        )
        return {
            "trained_model": text_classification_pipeline.outputs.mlflow_model_folder
        }

    # Create the pipeline object
    pipeline_object = create_pipeline()

    # Configure pipeline settings
    pipeline_object.settings.force_rerun = True
    pipeline_object.settings.continue_on_step_failure = False

    # Submit the pipeline job
    pipeline_job = workspace_ml_client.jobs.create_or_update(
        pipeline_object, experiment_name=experiment_name
    )

    # Wait for the pipeline job to complete
    workspace_ml_client.jobs.stream(pipeline_job.name)



if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    # sku_override = get_sku_override()
    # if sku_override is None:
    #     check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    # print values of all above variables
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['resource_group']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")
    print("Here is my test model name : ", test_model_name)
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    print("workspace_name : ", queue.workspace)
    workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    registry_ml_client = MLClient(credential, registry_name="azureml-preview-test1")
    experiment_name = "Auto_text-classification-emotion-detection"

    # # generating a unique timestamp that can be used for names and versions that need to be unique
    # timestamp = str(int(time.time()))

    # Define the compute cluster name and size
    compute_cluster = "gpu-cluster-big"
    compute_cluster_size = "Standard_NC6s_v3"
    
    # Optional: Define a list of allowed compute sizes (if any)
    computes_allow_list = ["standard_nc6s_v3", "standard_nc12s_v2"]
    
    # Call the function
    compute, gpus_per_node, compute_cluster = create_or_get_aml_compute(workspace_ml_client, compute_cluster, compute_cluster_size, computes_allow_list)




    
    # compute = create_or_get_compute_target(workspace_ml_client, queue.compute)
    print("printing:",{compute})
    env_list = workspace_ml_client.environments.list(name=queue.environment)
    latest_version = 0
    for env in env_list:
        if latest_version <= int(env.version):
            latest_version = int(env.version)
    print("Latest Environment Version:", latest_version)
    latest_env = workspace_ml_client.environments.get(
        name=queue.environment, version=str(latest_version))
    print("Latest Environment :", latest_env)
    version_list = list(workspace_ml_client.models.list(test_model_name))

    client = MlflowClient()
    
    #foundation_model, foundation_model_name = get_latest_model_version(workspace_ml_client, test_model_name.lower())
    foundation_model = get_latest_model_version(workspace_ml_client, test_model_name.lower())
    training_parameters, optimization_parameters = get_training_and_optimization_parameters(foundation_model)
    #gpus_per_node = find_gpus_in_compute(workspace_ml_client, compute)
    print(f"Number of GPUs in compute: {gpus_per_node}")
    pipeline_job = create_and_run_azure_ml_pipeline(foundation_model, compute_cluster, gpus_per_node, training_parameters, optimization_parameters, experiment_name)
    print("Completed")

    
    
    # registered_model_detail = client.get_latest_versions(
    #     name=test_model_name, stages=["None"])
    # model_detail = registered_model_detail[0]
    
    # print("Latest registered model version is : ", model_detail.version)
    
    # loaded_model = mlflow.transformers.load_model(model_uri=model_detail.source, return_type="pipeline")
    # model_source_uri = foundation_model.properties["mlflow.modelSourceUri"]
    # print("model_source_uri---------------------",model_source_uri)
    # loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri)
    # LM=load_model(model_detail)
    # print("LM-----------------------------",LM)
    # environment_variables = {"test_model_name": test_model_name
    #                         ,"model_source_uri": model_detail.source}
    # print("environment_variables-------------",environment_variables)
    # print("queue.compute---",queue.compute)
    # print("queue.workspace====",queue.workspace)
    # command_job = run_azure_ml_job(code="./", command_to_run="python FTTest.py",
    #                                environment=latest_env, compute=queue.compute, environment_variables=environment_variables)
    
    # create_and_get_job_studio_url(command_job, workspace_ml_client)

    # FT_model_name = f"FT_P-TC-{test_model_name}"
   

    # InferenceAndDeployment = ModelInferenceAndDeployemnt(
    #     test_model_name=FT_model_name,
    #     workspace_ml_client=workspace_ml_client,
    #     registry=queue.registry
    # )
    # InferenceAndDeployment.model_infernce_and_deployment(
    #     instance_type=queue.instance_type
    # )









































# def create_and_submit_pipeline(
#     ml_client: MLClient,
#     registry_ml_client,
#     foundation_model,
#     compute_cluster,
#     experiment_name,
#     gpus_per_node,
#     training_parameters,
#     optimization_parameters,
# ):
#     # Fetch the pipeline component
#     pipeline_component_func = registry_ml_client.components.get(
#         name="text_classification_pipeline", label="latest"
#     )

#     # Define the pipeline job
#     @pipeline()
#     def create_pipeline():
#         text_classification_pipeline = pipeline_component_func(
#             mlflow_model_path=foundation_model.id,
#             compute_model_import=compute_cluster,
#             compute_preprocess=compute_cluster,
#             compute_finetune=compute_cluster,
#             compute_model_evaluation=compute_cluster,
#             train_file_path=Input(
#                 type="uri_file", path="./emotion-dataset/small_train.jsonl"
#             ),
#             validation_file_path=Input(
#                 type="uri_file", path="./emotion-dataset/small_validation.jsonl"
#             ),
#             test_file_path=Input(
#                 type="uri_file", path="./emotion-dataset/small_test.jsonl"
#             ),
#             evaluation_config=Input(
#                 type="uri_file", path="./text-classification-config.json"
#             ),
#             sentence1_key="text",
#             label_key="label_string",
#             number_of_gpu_to_use_finetuning=gpus_per_node,
#             **training_parameters,
#             **optimization_parameters
#         )
#         return {
#             "trained_model": text_classification_pipeline.outputs.mlflow_model_folder
#         }

#     pipeline_object = create_pipeline()
#     pipeline_object.settings.force_rerun = True
#     pipeline_object.settings.continue_on_step_failure = False

#     # Submit the pipeline job
#     pipeline_job = ml_client.jobs.create_or_update(
#         pipeline_object, experiment_name=experiment_name
#     )

#     # Wait for the pipeline job to complete
#     ml_client.jobs.stream(pipeline_job.name)
#     return pipeline_job

# # Usage example:
# # Replace these values with your actual inputs
# ml_client = MLClient(...)  # Initialize your MLClient
# registry_ml_client = ...  # Initialize your registry_ml_client
# foundation_model = ...  # Your foundation model
# compute_cluster = ...  # Your compute cluster
# experiment_name = "your_experiment_name"
# gpus_per_node = ...  # Number of GPUs per node
# training_parameters = ...  # Your training parameters
# optimization_parameters = ...  # Your optimization parameters

# pipeline_job = create_and_submit_pipeline(
#     ml_client,
#     registry_ml_client,
#     foundation_model,
#     compute_cluster,
#     experiment_name,
#     gpus_per_node,
#     training_parameters,
#     optimization_parameters,
# )
