import requests
from azureml.core import Workspace, Environment
from io import StringIO

# URLs for raw content
CONDA_YAML_URL = 'https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/conda.yml'
CONFIG_JSON_URL ='https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/config.json'

# Fetch conda.yaml content
response = requests.get(CONDA_YAML_URL)
response.raise_for_status()
conda_content = StringIO(response.text)

# Fetch config.json content and connect to Azure ML workspace
response = requests.get(CONFIG_JSON_URL)
response.raise_for_status()
config_content = response.json()
# Connect to Azure ML Workspace using the fetched config.json

 subscription_id = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
 resource_group = "huggingface-registry-test1"
 workspace_name = "test-eastus"

ws = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name
)


# Create and register the environment using the fetched conda.yaml
env_name = "automate-create-env"  
testenv = Environment.from_stream(workspace=ws, name=env_name, stream=conda_content)
testenv.register(workspace=ws)
