import os
from azureml.core import Workspace, Environment
from azureml.core.authentication import ServicePrincipalAuthentication

# Retrieve environment variables
tenant_id = os.environ['AZURE_TENANT_ID']
client_id = os.environ['AZURE_CLIENT_ID']
client_secret = os.environ['AZURE_CLIENT_SECRET']
subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']

# Service principal authentication
sp_auth = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret
)

# Connect to existing Azure ML Workspace
subscription_id = "80c77c76-74ba-4c8c-8229-4c3b2957990c"
resource_group = "huggingface-registry-test1"
workspace_name = "test-eastus"

ws = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name
)

# Create and register the environment
env_name = "automate-create-env"

env = Environment.from_conda_specification(
    name=env_name,
    file_path="https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/conda.yml"  

env.register(workspace=ws)
