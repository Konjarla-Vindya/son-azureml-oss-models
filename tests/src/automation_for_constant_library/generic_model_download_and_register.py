from transformers import AutoModel,AutoTokenizer
#import transformers
from azureml.core import Workspace
#from azureml.core import Workspace
#from azureml.mlflow import get_mlflow_tracking_uri
import os 
import mlflow
test_model_name = os.environ.get('test_model_name')
subscription = os.environ.get('subscription')
resource_group = os.environ.get('resource_group')
workspace_name = os.environ.get('workspace')
workspace_obj = os.environ.get("workspcae_object")
class Model:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
    
    def download_model_and_tokenizer(self)->dict:
        model = AutoModel.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_and_tokenizer = {"model":model, "tokenizer":tokenizer}
        return model_and_tokenizer
    
    def register_model_in_workspace(self, model_and_tokenizer, workspace):
        #task = self.queue.models[self.model_name].task
        artifact_path = self.model_name + "_artifact"
        registered_model_name = self.model_name + "_registered"
        mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        mlflow.transformers.log_model(
            transformers_model = model_and_tokenizer,
            #task=task,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def download_and_register_model(self)->dict :
        model_and_tokenizer = self.download_model_and_tokenizer()
        # workspace = Workspace(
        #         subscription_id = subscription,
        #         resource_group = resource_group,
        #         workspace_name = workspace_name
        #     )
        self.register_model_in_workspace(model_and_tokenizer, workspace=workspace_obj)
        return model_and_tokenizer

if __name__ == "__main__":
    model = Model(model_name=test_model_name)
    model.download_and_register_model()

    