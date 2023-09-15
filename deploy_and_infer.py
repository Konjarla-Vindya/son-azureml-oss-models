import json
import time
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    ModelConfiguration,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer,
)
from box import ConfigBox

# Load configuration from config.json
with open("credentials.json", "r") as config_file:
    config = json.load(config_file)

# Define your configuration
config_box = ConfigBox(config)

# Extract parameters from the configuration
model_name = config_box.model_name
model_version = config_box.model_version
question = config_box.question
context = config_box.context
subscription_id = config_box.subscription_id
resource_group = config_box.resource_group
workspace_name = config_box.workspace_name
online_endpoint_name_prefix = config_box.online_endpoint_name_prefix
deployment_name = config_box.deployment_name
instance_type = config_box.instance_type
instance_count = config_box.instance_count
initial_delay = config_box.initial_delay
traffic_percentage = config_box.traffic_percentage

# Initialize Azure ML Workspace
credential = DefaultAzureCredential()
ws = Workspace(subscription_id, resource_group, workspace_name)

# Fetch the registered model
version_list = list(ws.models.list(model_name))
foundation_model = ""
if len(version_list) == 0:
    print("Model not found in registry")
else:
    model_version = version_list[0].version
    foundation_model = ws.models.get(model_name, model_version)
    print(
        f"Using model name: {foundation_model.name}, version: {foundation_model.version}, id: {foundation_model.id} for inferencing"
    )

print(f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")

# Load the registered model pipeline
model_source_uri = foundation_model.properties["mlflow.modelSourceUri"]
registered_model_pipeline = mlflow.transformers.load_model(model_uri=model_source_uri)

# Define a question-answering configuration
question_answering = ConfigBox(
    {
        "inputs": {
            "question": question,
            "context": context,
        }
    }
)

# Perform inference using the registered model pipeline
inference_result = registered_model_pipeline(question_answering.inputs)

print("Inference Result:")
print(inference_result)

# Create an online endpoint
timestamp = int(time.time())
online_endpoint_name = online_endpoint_name_prefix + str(timestamp)
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for " + foundation_model.name + ", qa",
    auth_mode="key",
)
ws.ml.endpoints.create_or_update(endpoint).wait()

# Create a deployment
demo_deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type=instance_type,
    instance_count=instance_count,
    liveness_probe=ProbeSettings(initial_delay=initial_delay),
)
ws.online_deployments.create_or_update(demo_deployment).wait()
endpoint.traffic = {deployment_name: traffic_percentage}
ws.endpoints.create_or_update(endpoint).result()

print(f"Online endpoint '{online_endpoint_name}' deployed with deployment '{deployment_name}'")
