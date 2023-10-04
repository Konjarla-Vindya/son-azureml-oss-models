from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from utils.logging import get_logger
import mlflow
import time
import sys


COMPUTE_CLUSTER = "cpu-cluster"

logger = get_logger(__name__)
class AzurePipeline:
    def __init__(self, workspace_ml_client, registry_ml_client, task) -> None:
        self.workspace_ml_client = workspace_ml_client
        self.registry_ml_client = registry_ml_client
        self.task = task


    @pipeline()
    def evaluation_pipeline(self, mlflow_model, data_path):
        try:
            logger.info("Started configuring the job")
            pipeline_component_func = self.registry_ml_client.components.get(
                name="mlflow_oss_model_evaluation_pipeline", label="latest"
            )
            evaluation_job = pipeline_component_func(
                # specify the foundation model available in the azureml system registry or a model from the workspace
                # mlflow_model = Input(type=AssetTypes.MLFLOW_MODEL, path=f"{mlflow_model_path}"),
                mlflow_model=mlflow_model,
                # test data
                test_data=Input(type=AssetTypes.URI_FILE, path=data_path),
                # The following parameters map to the dataset fields
                input_column_names="input_string",
                label_column_name="ro",
                # compute settings
                compute_name=COMPUTE_CLUSTER,
                # specify the instance type for serverless job
                # instance_type= "STANDARD_NC24",
                # Evaluation settings
                task="text-translation",
                # config file containing the details of evaluation metrics to calculate
                evaluation_config=Input(
                    type=AssetTypes.URI_FILE, path="./evaluation/eval_config.json"),
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

    def run_pipeline(self, data_path, foundation_model):
        try:
            pipeline_jobs = []
            experiment_name = "text-translation-evaluation"
            pipeline_object = self.evaluation_pipeline(
                    mlflow_model=Input(type=AssetTypes.MLFLOW_MODEL, path=f"{foundation_model.id}"),
                    #mlflow_model = f"{foundation_model.id}",
                    data_path = data_path
                )
            # don't reuse cached results from previous jobs
            pipeline_object.settings.force_rerun = True
            pipeline_object.settings.default_compute = COMPUTE_CLUSTER

            # set continue on step failure to False
            pipeline_object.settings.continue_on_step_failure = False

            timestamp = str(int(time.time()))
            pipeline_object.display_name = f"eval-{foundation_model.name}-{timestamp}"
            pipeline_job = self.workspace_ml_client.jobs.create_or_update(
                pipeline_object, experiment_name=experiment_name
            )
            # add model['name'] and pipeline_job.name as key value pairs to a dictionary
            pipeline_jobs.append({"model_name": foundation_model.name, "job_name": pipeline_job.name})
            # wait for the pipeline job to complete
            self.workspace_ml_client.jobs.stream(pipeline_job.name)
            return pipeline_jobs
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
                         f" the exception is this one : \n {ex}")
            raise Exception(ex)
