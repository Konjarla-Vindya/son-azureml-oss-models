import os
from FT_initial_automation import load_model  
import mlflow
import tranformers
model_source_uri=os.environ.get('model_source_uri')
test_model_name = os.environ.get('test_model_name')
print("-----------------Im inside the python code")
print("test_model_name-----------------",test_model_name)
loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
print("loaded_model-----------------------------",loaded_model)
