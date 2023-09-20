import os
from FT_initial_automation import load_model  
model_source_uri=os.environ.get('model_source_uri')
test_model_name = os.environ.get('test_model_name')
print("-----------------Im inside the python code")
print("test_model_name-----------------",test_model_name)
LM=load_model(model_source_uri)
print("loaded_model-----------------------------",LM)
