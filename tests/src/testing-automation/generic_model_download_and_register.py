from transformers import AutoModel, AutoTokenizer, AutoConfig
import transformers
#from azureml.core import Workspace
#from azureml.core import Workspace
#from azureml.mlflow import get_mlflow_tracking_uri
from urllib.request import urlopen
from box import ConfigBox
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from mlflow.tracking.client import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from transformers import pipeline
import pandas as pd
import os
import mlflow
import re
import json


# import json
import json
# store the URL in url as
# parameter for urlopen
URL = "https://huggingface.co/api/models"
COLUMNS_TO_READ = ["modelId", "pipeline_tag", "tags"]
STRING_TO_CHECK = 'transformers'
FILE_NAME = "task_and_library.json"

test_model_name = os.environ.get('test_model_name')


class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def get_task(self) -> str:
        """ This method will read the huggin face api url data in a dataframe. Then it will findout 
        the model which is of transformer type . Then it will find that particular model and its task

        Returns:
            str: task name
        """
        response = urlopen(URL)
        # Load all the data with the help of json
        data_json = json.loads(response.read())
        # Convert it into dataframe and mention the specific column
        df = pd.DataFrame(data_json, columns=COLUMNS_TO_READ)
        # Find the data with the model which will be having trasnfomer tag
        df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]
        # Find the data with that particular name
        required_data = df[df.modelId.apply(lambda x: x == self.model_name)]
        # Get the task
        required_data = required_data["pipeline_tag"].to_string()
        pattern = r'[0-9\s+]'
        final_data = re.sub(pattern, '', required_data)
        return final_data

    def get_sample_input_data(self, task: str):
        """This method will load the sample input data based on the task name

        Args:
            task (str): task name

        Returns:
            _type_: _description_
        """
        scoring_file = f"sample_inputs/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
                print(f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")

        return scoring_input
    
    def get_library_to_load_model(self, task: str) -> str:
        """ Takes the task name and load the  json file findout the library 
        which is applicable for that task and retyrun it 

        Args:
            task (str): required the task name 
        Returns:
            str: return the library name
        """
        try:
            with open(FILE_NAME) as f:
                model_with_library = ConfigBox(json.load(f))
                print(f"scoring_input file:\n\n {model_with_library}\n\n")
        except Exception as e:
            print(
                f"::warning:: Could not find scoring_file: {model_with_library}. Finishing without sample scoring: \n{e}")
        return model_with_library.get(task)

    def download_model_and_tokenizer(self, task: str) -> dict:
        """" This method will download the model and tokenizer and return it in a 
        dictionary

        Args:
            task (str): task name

        Returns:
            dict: model and tokenizer
        """
        # model_detail = AutoConfig.from_pretrained(self.model_name)
        # res_dict = model_detail.to_dict().get("architectures")
        # if res_dict is not None:
        #     model_library_name = res_dict[0]
        # else:
        #     rare_model_dict = self.load_rare_model()
        #     model_library_name = rare_model_dict.get(self.model_name)
        #model_library_name = model_detail.to_dict()["architectures"][0]

        # Get the library name from this method from which we will load the model
        model_library_name = self.get_library_to_load_model(task=task)
        print("Library name is this one : ", model_library_name)
        # Load the library from the transformer
        model_library = getattr(transformers, model_library_name)
        # From the library load the model
        model = model_library.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_and_tokenizer = {"model": model, "tokenizer": tokenizer}
        return model_and_tokenizer

    def register_model_in_workspace(self, model_and_tokenizer: dict, scoring_input: ConfigBox, task: str, registered_model_name: str):
        """ I will load the pipeline with the model name if its a fill mask task then it will get the 
        masked token and convert the input to that model type . It will generate the model signature . 
        It will log and register the model with mlflow

        Args:
            model_and_tokenizer (dict): contains model and tokenizer
            scoring_input (ConfigBox): contains the data
            task (str): task name
        """
        # Load the transformer pipeline with the help of model and task
        model_pipeline = transformers.pipeline(
            task=task, model=model_and_tokenizer["model"], tokenizer=model_and_tokenizer["tokenizer"])
        # If the task is fill-mask then get the mask_token and replace the input data with that mask token
        if task == "fill-mask":
            pipeline_tokenizer = model_pipeline.tokenizer
            for index in range(len(scoring_input.input_data)):
                scoring_input.input_data[index] = scoring_input.input_data[index].replace(
                    "<mask>", pipeline_tokenizer.mask_token).replace("[MASK]", pipeline_tokenizer.mask_token)

        # Generate the transformer model output for that particular model
        output = generate_signature_output(
            model_pipeline, scoring_input.input_data)
        # It will infer the signature directly from input and output
        signature = infer_signature(scoring_input.input_data, output)

        artifact_path = registered_model_name + "-artifact"
        # With the help of mlflow log and register the model in the workspace
        mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            task=task,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=scoring_input.input_data
        )

    def download_and_register_model(self, task, scoring_input, registered_model_name) -> dict:
        """ This method will be controlling all execution of methods 

        Returns:
            dict: _description_
        """
        # Get the model and tokenizer
        model_and_tokenizer = self.download_model_and_tokenizer(task=task)
        # Register the model in the workspace
        self.register_model_in_workspace(
            model_and_tokenizer=model_and_tokenizer,
            scoring_input=scoring_input,
            task=task,
            registered_model_name=registered_model_name
        )
        return model_and_tokenizer

    def registered_model_inference(self, task, scoring_input, registered_model_name) -> None:
        """_summary_

        Args:
            task (_type_): _description_
            scoring_input (_type_): _description_
            registered_model_name (_type_): _description_
        """

        client = MlflowClient()
        registered_model_detail = client.get_latest_versions(
            name=registered_model_name, stages=["None"])
        model_detail = registered_model_detail[0]
        print("Latest registered model version is : ", model_detail.version)
        loaded_model_pipeline = mlflow.transformers.load_model(
            model_uri=model_detail.source, return_type="pipeline")
        
        #Alternate Approach
        # path = f"models:/{registered_model_name}/latest"
        # downloaded_path = ModelsArtifactRepository(path).download_artifacts(artifact_path="")
        # loaded_model_pipeline = mlflow.transformers.load_model(model_uri=downloaded_path, return_type="pipeline")

        if task == "fill-mask":
            pipeline_tokenizer = loaded_model_pipeline.tokenizer
            for index in range(len(scoring_input.input_data)):
                scoring_input.input_data[index] = scoring_input.input_data[index].replace(
                    "<mask>", pipeline_tokenizer.mask_token).replace("[MASK]", pipeline_tokenizer.mask_token)

        output = loaded_model_pipeline(scoring_input.input_data)
        print("My outupt is this : ", output)


if __name__ == "__main__":
    model = Model(model_name=test_model_name)
    # Get the sample input data
    task = model.get_task()
    # Get the sample input data
    scoring_input = model.get_sample_input_data(task=task)
    print("This is the task associated to the model : ", task)
    # If threr will be model namr with / then replace it
    registered_model_name = test_model_name.replace("/", "-")
    model.download_and_register_model(
        task=task, scoring_input=scoring_input, registered_model_name=registered_model_name)
    model.registered_model_inference(
        task=task, scoring_input=scoring_input, registered_model_name=registered_model_name)
    # workspace = Workspace.from_config()
    # print(workspace)
    # client = mlflow.tracking.MlflowClient()
    # result = client.get_registered_model(test_model_name)
    # print(result)
    # print("Type of result : ", type(result))
    # print("tags : ", str(result.tags))
    # registered_model = client.get_latest_versions(test_model_name, stages=["None"])
    # print("registered_model : ",registered_model)
    # print(" Type of registered_model : ", type(registered_model))
    #client.get_model_version(test_model_name, version=latest)
    # model = client.get_latest_versions(test_model_name, stages=None)
    # print(model)
