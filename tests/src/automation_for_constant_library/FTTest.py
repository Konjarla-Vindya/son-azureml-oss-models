import os
from FT_initial_automation import load_model  
test_model_name = os.environ.get('test_model_name')
print("-----------------Im inside the python code")
print("test_model_name-----------------",test_model_name)
print("loaded model-----------------------",loaded_model)
