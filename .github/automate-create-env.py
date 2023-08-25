import requests
from azureml.core import Workspace, Environment
from io import StringIO

# URLs for raw content
CONDA_YAML_URL = 'https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/conda.yml'
CONFIG_JSON_URL = 'https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/config.json'

# Fetch conda.yaml content
response = requests.get(CONDA_YAML_URL)
response.raise_for_status()
conda_content = StringIO(response.text)

# Fetch config.json content and connect to Azure ML workspace
response = requests.get(CONFIG_JSON_URL)
response.raise_for_status()
config_content = response.json()

# Connect to Azure ML Workspace using the fetched config.json
ws = Workspace.from_config(path=config_content)

# Create and register the environment using the fetched conda.yaml
env_name = "test_automate_env"  # You can change this to your preferred environment name
testenv = Environment.from_stream(workspace=ws, name=env_name, stream=conda_content)
testenv.register(workspace=ws)

