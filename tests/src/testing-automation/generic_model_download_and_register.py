from transformers import AutoModel,AutoTokenizer,AutoConfig
import transformers
#from azureml.core import Workspace
#from azureml.core import Workspace
#from azureml.mlflow import get_mlflow_tracking_uri
from urllib.request import urlopen
from box import ConfigBox
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from transformers import pipeline
import pandas as pd
import os 
import mlflow
import re

  
# import json
import json
# store the URL in url as 
# parameter for urlopen
url = "https://huggingface.co/api/models"
columns_to_read = ["modelId","pipeline_tag","tags"]
string_to_check = 'transformers'

test_model_name = os.environ.get('test_model_name')
subscription = os.environ.get('subscription')
resource_group = os.environ.get('resource_group')
workspace_name = os.environ.get('workspace')

class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
    
    def get_task_and_sample_data(self) -> pd.DataFrame:
        response = urlopen(url)
        data_json = json.loads(response.read())
        df = pd.DataFrame(data_json, columns=columns_to_read)
        df = df[df.tags.apply(lambda x: string_to_check in x)]
        required_data = df[df.modelId.apply(lambda x: x == self.model_name)]
        required_data = required_data["pipeline_tag"].to_string()
        pattern = r'[0-9\s+]'
        final_data = re.sub(pattern, '', required_data)
        return final_data
    
    def get_sample_input_data(self):
        final_data = self.get_task_and_sample_data()
        task = final_data
        print("task:",task)
        scoring_file = f"sample_inputs/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = ConfigBox(json.load(f))
                print (f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print (f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")
        
        return scoring_input, task

    def download_model_and_tokenizer(self)->dict:
        model_detail = AutoConfig.from_pretrained(self.model_name)
        model_library_name = model_detail.to_dict()["architectures"][0]
        model_library = getattr(transformers, model_library_name)
        model = model_library.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # config = AutoConfig.from_pretrained(self.model_name)
        # config_dict = config.to_dict()
        # task_dict = config_dict["task_specific_params"]
        # task=list(task_dict.keys())[0]

        model_and_tokenizer = {"model":model, "tokenizer":tokenizer}
        return model_and_tokenizer
    
    def register_model_in_workspace(self, model_and_tokenizer, sample_data, task):
        #task = self.queue.models[self.model_name].task
        model_pipeline = transformers.pipeline(task=task, model=self.model_name)
        output = generate_signature_output(model_pipeline, sample_data.inputs)
        signature = infer_signature(sample_data.inputs, output)
        artifact_path = self.model_name + "-artifact"
        registered_model_name = self.model_name
        # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        mlflow.transformers.log_model(
            transformers_model = model_and_tokenizer,
            task=task,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=sample_data.inputs
        )

    def download_and_register_model(self)->dict :
        model_and_tokenizer = self.download_model_and_tokenizer()
        sample_data, task = self.get_sample_input_data()
        self.register_model_in_workspace(model_and_tokenizer, sample_data, task)
        return model_and_tokenizer

if __name__ == "__main__":
    model = Model(model_name=test_model_name)
    model.download_and_register_model()
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
