from azureml.core import Workspace, Environment
import yaml

def create_conda_yaml(channels, conda_dependencies, pip_dependencies, env_name):
    # Define the environment
    environment = {
        'name': env_name,
        'channels': channels,
        'dependencies': conda_dependencies + [{'pip': pip_dependencies}]
    }

    # Write to a YAML file
    with open("conda.yaml", "w") as file:
        yaml.safe_dump(environment, file)
# Usage
channels = ["conda-forge", "defaults"]
conda_dependencies = ["python=3.10", "numpy", "pandas", "pip"]
pip_dependencies = [
   "azureml-mlflow","azureml-core","ipython","datasets","accelerate==0.21.0",
   "ipykernel","evaluate","azure-ai-ml","numpy","tensorflow==2.9","mlflow==2.4",
   "cffi==1.15.1","dill==0.3.6","google-api-core==2.11.0","ipython==8.8.0",
   "numpy==1.23.5","packaging==21.3","protobuf==3.20.3","pyyaml==6.0",
   "requests==2.28.2","safetensors==0.3.1","scikit-learn==1.2.2","scipy==1.10.1",
   "torch==2.0.1","torchvision==0.15.2","transformers==4.29.0",
   "xformers==0.0.20","azureml-mlflow==1.52.0","azure-core==1.27.1","torchvision",
   "sacremoses","python-box","sentencepiece","fugashi[unidic-lite]"
   ]
   
env_name = "auto_testenv"

create_conda_yaml(channels, conda_dependencies, pip_dependencies, env_name)

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


env_name.register(workspace=ws)

