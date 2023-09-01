from transformers import AutoModel, AutoTokenizer, AutoConfig
import transformers
from box import ConfigBox
from transformers import pipeline
import os
import mlflow
import json

foundation_model_uri = os.environ.get('foundation_model_uri')
task = os.environ.get("task")
#path = os.environ.get("path")
#registry = os.path.get("registry")

class ModelLodAndInference:
    def __init__(self, foundation_model_uri) -> None:
        self.foundation_model_uri = foundation_model_uri

    def get_sample_input_data(self, task:str):
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
    
    def model_load_and_inference(self, task):
        scoring_input = self.get_sample_input_data(task=task)
        loaded_model_pipeline = mlflow.transformers.load_model(
             model_uri=self.foundation_model_uri, return_type="pipeline")
        if task == "fill-mask":
            pipeline_tokenizer = loaded_model_pipeline.tokenizer
            for index in range(len(scoring_input.inputs)):
                scoring_input.inputs[index] = scoring_input.inputs[index].replace(
                    "<mask>", pipeline_tokenizer.mask_token).replace("[MASK]", pipeline_tokenizer.mask_token)

        output = loaded_model_pipeline(scoring_input.inputs)
        print("My outupt is this : ", output)

if __name__ == "main":
    model_load_and_inference = ModelLodAndInference(
        foundation_model_uri=foundation_model_uri
    )
    model_load_and_inference.model_load_and_inference(task=task)