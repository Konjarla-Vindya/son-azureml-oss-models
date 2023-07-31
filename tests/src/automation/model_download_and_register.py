#from transformers import AutoModelForSequenceClassification,AutoTokenizer
import transformers
#from azureml.core import Workspace
#from azureml.mlflow import get_mlflow_tracking_uri
import mlflow

class Model:
    def __init__(self, model_name, queue) -> None:
        self.model_name = model_name
        self.queue = queue
    
    def download_model_and_tokenizer(self)->dict:
        model_library = self.queue.models[self.model_name].model_library
        tokenizer_library = self.queue.models[self.model_name].tokenizer_library
        model = transformers.model_library.from_pretrained(self.model_name)
        tokenizer = transformers.tokenizer_library.from_pretrained(self.model_name)
        model_and_tokenizer = {"model":model, "tokenizer":tokenizer}
        return model_and_tokenizer
    
    def register_model_in_workspace(self, model_and_tokenizer, workspace):
        task = self.queue.models[self.model_name].task
        artifact_path = self.model_name + "_artifact"
        registered_model_name = self.model_name + "_registered"
        # ws = Workspace(
        #         subscription_id = self.queue.subscription,
        #         resource_group_name = self.queue.resource_group,
        #         workspace_name = self.queue.workspace
        #     )
        mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        mlflow.transformers.log_model(
            transformers_model = model_and_tokenizer,
            task=task,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def download_and_register_model(self, workspace)->dict :
        model_and_tokenizer = self.download_model_and_tokenizer()
        self.register_model_in_workspace(model_and_tokenizer, workspace)
        return model_and_tokenizer

    