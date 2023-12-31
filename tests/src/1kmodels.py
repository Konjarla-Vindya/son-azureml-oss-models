print("py started")
from urllib.request import urlopen

import os
import yaml
import json
import pandas as pd
url = "https://huggingface.co/api/models"
response = urlopen(url)
data_json = json.loads(response.read())
#print(data_json)
df=pd.DataFrame(data_json)
def contains(x):
    return 'transformers' in x
df["transformers"]=df["tags"].apply(contains)

list1=["fill-mask","translation","text-generation","token-classification","summarization","text-classification","question-answering"]
df3=df.loc[df["pipeline_tag"].isin(list1),["modelId","downloads","transformers"]].sort_values(by="downloads",ascending=False)
df4=df3[df3["transformers"]==True]
df4.head()
#df2=df.loc[df["pipeline_tag"].isin(list1),["id","pipeline_tag","downloads"]].sort_values(by="downloads",ascending=False).head(1000)
model_names=df4["modelId"].head(2)
#model_names

# import os
# print(os.getcwd())
# base_dir = ".github/workflows"
# file_name = "1kmodels.yml"
# print("fetched value")
# file_path = os.path.join(base_dir, file_name)


# The path to the original YAML file
original_yaml_file = "../../.github/workflows/1kmodels.yml"
#.github/workflows/1kmodels.yml

# The directory where the generated YAML files will be stored
output_directory = "../../.github/workflows/1kmodels"
print("Current working directory:", os.getcwd())
print("Output directory:", output_directory)
os.makedirs(output_directory, exist_ok=True)
print("Output directory created:", os.path.exists(output_directory))

# List of 1000 model names
#model_names = ['model1', 'model2', ..., 'model1000']

def generate_workflow_yaml(model_name):
    #with open(file_path, 'r') as f:
    with open(original_yaml_file, 'r') as f:
        yaml_content = f.read()

    # Modify the 'test_model_name' and 'test_queue' fields for the current model
    yaml_content = yaml_content.replace('name: bert-base-uncased', f'name: {model_name}')
    yaml_content = yaml_content.replace('test_model_name: bert-base-uncased', f'test_model_name: {model_name}')
    yaml_content = yaml_content.replace('test_queue: model_with_static_library', f'test_queue: {model_name}_queue')

    # Create a filename for the generated YAML file based on the model name
    generated_yaml_filename = f'{model_name}_workflow.yaml'
    print("My Generated YAML filename:", generated_yaml_filename)

    # Write the modified YAML content to the generated YAML file
    generated_yaml_path = os.path.join(output_directory, generated_yaml_filename)
    print("Writing to:", generated_yaml_path)
    with open(generated_yaml_path, 'w') as f:
        f.write(yaml_content)
        print("Content written to file:")
        print(yaml_content)
        print("code parsed")

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Generate a YAML file for each model
    for model_name in model_names:
        generate_workflow_yaml(model_name)

    print("Generated YAML files for all 1000 models.")
