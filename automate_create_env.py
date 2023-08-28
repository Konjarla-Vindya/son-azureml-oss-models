import os
import requests
from azureml.core import Workspace, Environment

# Define your Azure ML workspace details
subscription_id = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
resource_group = "huggingface-registry-test1"
workspace_name = "test-eastus"

# URLs for GitHub raw content
CONDA_YAML_URL = 'https://raw.githubusercontent.com/Konjarla-Vindya/son-azureml-oss-models/main/.github/conda.yml'

# Download conda.yaml file from GitHub
response = requests.get(CONDA_YAML_URL)
response.raise_for_status()
with open('conda.yml', 'wb') as f:
    f.write(response.content)

# Connect to Azure ML Workspace
ws = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name
)

# Create and register Azure ML Environment from conda.yaml
env_name = "automate-create-env"
env = Environment.from_conda_specification(
    name=env_name,
    file_path="conda.yml"
)
env.register(workspace=ws)

print(f"Environment {env_name} has been created and registered.")
