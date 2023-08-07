import os

# The path to the original YAML file
original_yaml_file = 'https://github.com/Konjarla-Vindya/son-azureml-oss-models/new/main/.github/workflows'

# The directory where the generated YAML files will be stored
output_directory = '../path/to/generated_workflows/'

# List of 1000 model names
model_names = ['albert-base-v1',
 'albert-base-v2',
 'albert-large-v1',
 'albert-large-v2',
 'albert-xlarge-v1',
 'albert-xlarge-v2',
 'albert-xxlarge-v1',
 'albert-xxlarge-v2',
 'bert-base-cased-finetuned-mrpc',
 'bert-base-cased']

def generate_workflow_yaml(model_name):
    with open(original_yaml_file, 'r') as f:
        yaml_content = f.read()

    # Modify the 'test_model_name' and 'test_queue' fields for the current model
    yaml_content = yaml_content.replace('test_model_name: bert-base-uncased', f'test_model_name: {model_name}')
    yaml_content = yaml_content.replace('test_queue: model_with_static_library', f'test_queue: {model_name}_queue')

    # Create a filename for the generated YAML file based on the model name
    generated_yaml_filename = f'{model_name}_workflow.yaml'

    # Write the modified YAML content to the generated YAML file
    generated_yaml_path = os.path.join(output_directory, generated_yaml_filename)
    with open(generated_yaml_path, 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Generate a YAML file for each model
    for model_name in model_names:
        generate_workflow_yaml(model_name)

    print("Generated YAML files for all 1000 models.")
