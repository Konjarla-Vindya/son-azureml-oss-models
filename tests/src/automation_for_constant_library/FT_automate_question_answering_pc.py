import os
import time
import json
import sys
import mlflow
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from azure.ai.ml import command
from box import ConfigBox
from mlflow.tracking.client import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.dsl import pipeline
from azureml.core import Workspace, Environment
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

check_override = True
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
# the queue file also contains the details of the workspace, registry, subscription, resource group
def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
    
# this is useful if you want to force a specific sku for a model
def get_sku_override():
    try:
        with open('../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None
    
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
# Training parameters
def get_training_and_optimization_parameters(foundation_model):
    # Training parameters
    training_parameters = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 2e-1,
        "metric_for_best_model": "exact",
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

#Download dataset

def download_and_process_dataset():
    # Download the dataset using the helper script.
    exit_status = os.system("python ./download-dataset1.py --download_dir squad-dataset")
    if exit_status != 0:
        raise Exception("Error downloading dataset")

    # Load the train.jsonl, validation.jsonl, and test.jsonl files.
    train_df = pd.read_json("./squad-dataset/train.jsonl", lines=True)
    validation_df = pd.read_json("./squad-dataset/validation.jsonl", lines=True)
    #test_df = pd.read_json("./squad-dataset/test.jsonl", lines=True)

    # Set the fraction parameter to control the number of examples to be saved.
    frac = 1  # You can adjust this value as needed.

    # Save a fraction of the rows from the dataframes with a "small_" prefix in the ./wmt16-en-ro-dataset folder.
    train_df.sample(frac=frac).to_json("./squad-dataset/small_train.jsonl", orient="records", lines=True)
    validation_df, test_df = (
    validation_df[: len(validation_df) // 2],
    validation_df[len(validation_df) // 2 :],
    )
    validation_df.sample(frac=frac).to_json("./squad-dataset/small_validation.jsonl", orient="records", lines=True)
    test_df.sample(frac=frac).to_json("./squad-dataset/small_test.jsonl", orient="records", lines=True)
    train_df=train_df.iloc[:100,:]
    validation_df=validation_df.iloc[:100,:]
    test_df=test_df.iloc[:100,:]

# Example usage:
download_and_process_dataset()


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
        name="question_answering_pipeline_for_oss", label="latest"
    )
    # Model Training and Pipeline Setup
    @pipeline()
    def create_pipeline():
        translation_pipeline = pipeline_component_func(
            mlflow_model_path=foundation_model.id,
            # huggingface_id = 't5-small', # if you want to use a huggingface model, uncomment this line and comment the above line
            compute_model_import=compute_cluster,
            compute_preprocess=compute_cluster,
            compute_finetune=compute_cluster,
            compute_model_evaluation=compute_cluster,
            # map the dataset splits to parameters
            train_file_path=Input(
            type="uri_file", path="./squad-dataset/small_train.jsonl"
            ),
            validation_file_path=Input(
            type="uri_file", path="./squad-dataset/small_validation.jsonl"
            ),
           test_file_path=Input(type="uri_file", path="./squad-dataset/small_test.jsonl"),
           evaluation_config=Input(
            type="uri_file", path="./question-answering-config.json"
            ),
            #evaluation_config=Input(type="uri_file", path="./translation-config.json"),
           question_key="question",
           context_key="context",
           answers_key="answers",
           answer_start_key="answer_start", 
           answer_text_key="text",
            # training settings
           number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
            **training_parameters,
            **optimization_parameters
        )
        return {
            # map the output of the fine tuning job to the output of the pipeline job so that we can easily register the fine tuned model
            # registering the model is required to deploy the model to an online or batch endpoint
            "trained_model": question_answering_pipeline.outputs.mlflow_model_folder
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
    experiment_name = "question-answering-extractive-qna"
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



# def get_runs_from_mlflow(workspace_ml_client, experiment_name, pipeline_job):

#     mlflow_tracking_uri = workspace_ml_client.workspaces.get(
#         workspace_ml_client.workspace_name
#     ).mlflow_tracking_uri
#     mlflow.set_tracking_uri(mlflow_tracking_uri)

#     # Concatenate 'tags.mlflow.rootRunId=' and pipeline_job.name in single quotes as filter variable
#     filter = "tags.mlflow.rootRunId='" + pipeline_job.name + "'"
#     runs = mlflow.search_runs(
#         experiment_names=[experiment_name], filter_string=filter, output_format="list"
#     )
#     training_run = None
#     evaluation_run = None

#     # Get the training and evaluation runs.
#     # Using a workaround until the issue 'Bug 2320997: not able to show eval metrics in FT notebooks - mlflow client now showing display names' is fixed
#     for run in runs:
#         # Check if run.data.metrics.epoch exists
#         if "epoch" in run.data.metrics:
#             training_run = run
#         # Else, check if run.data.metrics.accuracy exists
#         elif "bleu_1" in run.data.metrics:
#             evaluation_run = run
#     if training_run:
#         print("Training metrics:\n")
#         print(json.dumps(training_run.data.metrics, indent=2))
#     else:
#         print("No Training job found")

#     if evaluation_run:
#         print("\nEvaluation metrics:\n")
#         print(json.dumps(evaluation_run.data.metrics, indent=2))
#     else:
#         print("No Evaluation job found")

#     return training_run, evaluation_run



# def register_model_from_pipeline_output(workspace_ml_client, pipeline_job, model_name, timestamp):
#     # Check if the `trained_model` output is available
#     print("Pipeline job outputs: ", workspace_ml_client.jobs.get(pipeline_job.name).outputs)

#     # Fetch the model from pipeline job output - not working, hence fetching from fine-tune child job
#     model_path_from_job = "azureml://jobs/{0}/outputs/{1}".format(
#         pipeline_job.name, "trained_model"
#     )

#     finetuned_model_name = model_name + "-wmt16-en-ro-src"
#     finetuned_model_name = finetuned_model_name.replace("/", "-")
#     print("Path to register model: ", model_path_from_job)

#     prepare_to_register_model = Model(
#         path=model_path_from_job,
#         type=AssetTypes.MLFLOW_MODEL,
#         name=finetuned_model_name,
#         version=timestamp,  # Use timestamp as the version to avoid version conflicts
#         description=model_name + " fine-tuned model for translation wmt16 en to ro",
#     )
#     print("Prepare to register model:\n", prepare_to_register_model)

#     # Register the model from the pipeline job output
#     registered_model = workspace_ml_client.models.create_or_update(prepare_to_register_model)
#     print("Registered model:\n", registered_model)

# # Call the function with the appropriate arguments to register the model
# register_model_from_pipeline_output(workspace_ml_client, pipeline_job, model_name, timestamp)
