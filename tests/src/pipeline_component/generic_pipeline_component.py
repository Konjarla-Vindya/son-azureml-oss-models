from azureml.core import Workspace
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import MLClient
import mlflow
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
from fetch_model_detail import ModelDetail
from dataset_loader import LoadDataset
from azureml_pipeline import AzurePipeline
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
import time
import pandas as pd
from fetch_task import HfTask


# constants
check_override = True

logger = get_logger(__name__)


# model to test
test_model_name = os.environ.get('test_model_name')


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

FILE_NAME = "pipeline_task.json"


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model


# finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    logger.info("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    check_mlflow_model = test_model_name
    index = model_list.index(check_mlflow_model)
    #index = model_list.index(test_model_name)
    logger.info(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            logger.warning("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        logger.info(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def create_or_get_compute_target(ml_client,  compute, instance):
    cpu_compute_target = compute
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except Exception:
        logger.info("Creating a new cpu compute target...")
        compute = AmlCompute(
            name=cpu_compute_target, size=instance, min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()

    return compute


def get_file_path(task):
    file_name = task+".json"
    data_path = f"./datasets/{task}{file_name}"
    return data_path


def get_dataset(task, data_path, latest_model):
    load_dataset = LoadDataset(
        task=task, data_path=data_path, latest_model=latest_model)
    task = task.replace("-", "_")
    # if task.__contains__("translation"):
    #     attribute = getattr(LoadDataset, "translation")
    # else:
    #     attribute = getattr(LoadDataset, task)
    attribute = getattr(LoadDataset, task)
    return attribute(load_dataset)


def get_pipeline_task(task):
    try:
        with open(FILE_NAME) as f:
            pipeline_task = ConfigBox(json.load(f))
            logger.info(
                f"Library name based on its task :\n\n {pipeline_task}\n\n")
    except Exception as e:
        logger.error(
            f"::Error:: Could not find library from here :{pipeline_task}.Here is the exception\n{e}")
    return pipeline_task.get(task)


# queue = get_test_queue()

# try:
#     credential = DefaultAzureCredential()
#     credential.get_token("https://management.azure.com/.default")
# except Exception as ex:
#     # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
#     credential = InteractiveBrowserCredential()
#     logger.info(f"workspace_name : {queue.workspace}")
# try:
#     workspace_ml_client = MLClient.from_config(credential=credential)
# except:
#     workspace_ml_client = MLClient(
#         credential=credential,
#         subscription_id=queue.subscription,
#         resource_group_name=queue.resource_group,
#         workspace_name=queue.workspace
#     )
# ws = Workspace(
#     subscription_id=queue.subscription,
#     resource_group=queue.resource_group,
#     workspace_name=queue.workspace
# )
# registry_ml_client = MLClient(
#     credential=credential,
#     # subscription_id="4f26493f-21d2-4726-92ea-1ddd550b1d27",
#     # resource_group_name="registry-builtin-prp-test",
#     registry_name="azureml-preview-test1"
# )
# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
COMPUTE_CLUSTER = "cpu-cluster"

registry_ml_client = ''
@pipeline()
def evaluation_pipeline(task, mlflow_model, test_data, input_column_names, label_column_name, evaluation_file_path, compute):
    try:
        logger.info("Started configuring the job")
        #data_path = "./datasets/translation.json"
        pipeline_component_func = registry_ml_client.components.get(
            name="mlflow_oss_model_evaluation_pipeline", label="latest"
        )
        evaluation_job = pipeline_component_func(
            # specify the foundation model available in the azureml system registry or a model from the workspace
            # mlflow_model = Input(type=AssetTypes.MLFLOW_MODEL, path=f"{mlflow_model_path}"),
            mlflow_model=mlflow_model,
            # test data
            test_data=Input(type=AssetTypes.URI_FILE, path=test_data),
            # The following parameters map to the dataset fields
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            # compute settings
            compute_name=compute,
            # specify the instance type for serverless job
            # instance_type= "STANDARD_NC24",
            # Evaluation settings
            task=task,
            # config file containing the details of evaluation metrics to calculate
            # evaluation_config=Input(
            #     type=AssetTypes.URI_FILE, path="./evaluation/eval_config.json"),
            evaluation_config=Input(
                type=AssetTypes.URI_FILE, path=evaluation_file_path),
            # config cluster/device job is running on
            # set device to GPU/CPU on basis if GPU count was found
            device="auto",
        )
        return {"evaluation_result": evaluation_job.outputs.evaluation_result}
    except Exception as ex:
        _, _, exc_tb = sys.exc_info()
        logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                     f" the exception is this one : \n {ex}")
        raise Exception(ex)


def display_metrics(pipeline_jobs):
    metrics_df = pd.DataFrame()
    for job in pipeline_jobs:
        logger.info(job)
        logger.info(job['model_name'][24:])
        # concat 'tags.mlflow.rootRunId=' and pipeline_job.name in single quotes as filter variable
        filter = "tags.mlflow.rootRunId='" + job["job_name"] + "'"
        runs = mlflow.search_runs(
            experiment_names=[
                experiment_name], filter_string=filter, output_format="list"
        )
        # get the compute_metrics runs.
        # using a hacky way till 'Bug 2320997: not able to show eval metrics in FT notebooks - mlflow client now showing display names' is fixed
        for run in runs:
            if len(run.data.metrics) > 0:
                print(run.data.metrics)
            print(run.data.metrics)
            # else, check if run.data.metrics.accuracy exists
            if "exact_match" in run.data.metrics:
                # get the metrics from the mlflow run
                run_metric = run.data.metrics
                # add the model name to the run_metric dictionary
                run_metric["model_name"] = job["model_name"]
                # convert the run_metric dictionary to a pandas dataframe
                temp_df = pd.DataFrame(run_metric, index=[0])
                # concat the temp_df to the metrics_df
                metrics_df = pd.concat(
                    [metrics_df, temp_df], ignore_index=True)
    return metrics_df


if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)
    # print values of all above variables
    logger.info(f"test_subscription_id: {queue['subscription']}")
    logger.info(f"test_resource_group: {queue['subscription']}")
    logger.info(f"test_workspace_name: {queue['workspace']}")
    logger.info(f"test_model_name: {test_model_name}")
    logger.info(f"test_registry: {queue['registry']}")
    logger.info(f"test_trigger_next_model: {test_trigger_next_model}")
    logger.info(f"test_queue: {test_queue}")
    logger.info(f"test_set: {test_set}")
    #logger.info(f"Here is my test model name : {test_model_name}")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    logger.info(f"workspace_name : {queue.workspace}")
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    logger.info(f"workspace_ml_client : {workspace_ml_client}")
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    registry_ml_client = MLClient(
        credential=credential,
        # subscription_id="4f26493f-21d2-4726-92ea-1ddd550b1d27",
        # resource_group_name="registry-builtin-prp-test",
        registry_name=queue.registry
    )
    logger.info(f"registry_ml_client : {registry_ml_client}")
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    compute_target = create_or_get_compute_target(
        ml_client = workspace_ml_client, compute=queue.compute, instance=queue.instance_type)
    # environment_variables = {
    #     "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT": 600.0, "test_model_name": test_model_name}
    # env_list = workspace_ml_client.environments.list(name=queue.environment)
    # latest_version = 0
    # for env in env_list:
    #     if latest_version <= int(env.version):
    #         latest_version = int(env.version)
    # logger.info(f"Latest Environment Version: {latest_version}")
    # latest_env = workspace_ml_client.environments.get(
    #     name=queue.environment, version=str(latest_version))
    # logger.info(f"Latest Environment : {latest_env}")
    # command_job = run_azure_ml_job(code="./", command_to_run="python generic_model_download_and_register.py",
    #                                environment=latest_env, compute=queue.compute, environment_variables=environment_variables)
    # create_and_get_job_studio_url(command_job, workspace_ml_client)
    model_detail = ModelDetail(workspace_ml_client=workspace_ml_client)
    latest_model = model_detail.get_model_detail(
        test_model_name=test_model_name)
    task = HfTask(model_name=test_model_name).get_task()
    data_path = get_file_path(task=task)
    input_column_names, label_column_name = get_dataset(task=task, data_path=data_path,
                                                        latest_model=latest_model)
    pieline_task = get_pipeline_task(task)
    # for key in pieline_task_dict.keys():
    #     if task.__contains__(key):
    #         pieline_task = pieline_task_dict.get(key)

    # azure_pipeline = AzurePipeline(
    #     workspace_ml_client=workspace_ml_client,
    #     registry_ml_client=registry_ml_client,
    #     task=pieline_task
    # )
    # pipeline_jobs = azure_pipeline.run_pipeline(
    #     data_path=data_path, foundation_model=latest_model)
    try:
        pipeline_jobs = []
        experiment_name = f"{pieline_task}-evaluation"
        pipeline_object = evaluation_pipeline(
            task=pieline_task,
            mlflow_model=Input(type=AssetTypes.MLFLOW_MODEL,
                               path=f"{latest_model.id}"),
            test_data=Input(type=AssetTypes.URI_FILE, path=data_path),
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            evaluation_file_path=Input(
                type=AssetTypes.URI_FILE, path=f"./evaluation/{task}/eval_config.json"),
            compute=queue.compute,
            #mlflow_model = f"{latest_model.id}",
            #data_path = data_path
        )
        # don't reuse cached results from previous jobs
        pipeline_object.settings.force_rerun = True
        pipeline_object.settings.default_compute = COMPUTE_CLUSTER

        # set continue on step failure to False
        pipeline_object.settings.continue_on_step_failure = False

        timestamp = str(int(time.time()))
        pipeline_object.display_name = f"eval-{latest_model.name}-{timestamp}"
        pipeline_job = workspace_ml_client.jobs.create_or_update(
            pipeline_object, experiment_name=experiment_name
        )
        # add model['name'] and pipeline_job.name as key value pairs to a dictionary
        pipeline_jobs.append(
            {"model_name": latest_model.name, "job_name": pipeline_job.name})
        # wait for the pipeline job to complete
        workspace_ml_client.jobs.stream(pipeline_job.name)
        # return pipeline_jobs
    except Exception as ex:
        _, _, exc_tb = sys.exc_info()
        logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                     f" the exception is this one : \n {ex}")
        raise Exception(ex)
    metrics_df = display_metrics(pipeline_jobs=pipeline_jobs)
    logger.info(f"Evaluation result {metrics_df}")
