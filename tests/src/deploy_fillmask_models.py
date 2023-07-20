from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
    AzureCliCredential,
)
from azure.ai.ml.entities import AmlCompute
import time

# try:
#    credential = AzureCliCredential()
#    credential.get_token("https://management.azure.com/.default")
try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    # connect to workspace
workspace_ml_client = MLClient(
    credential,
    subscription_id="80c77c76-74ba-4c8c-8229-4c3b2957990c",
    resource_group_name="sonata-test-rg",
    workspace_name="sonata-test-ws",
)

# the models, fine tuning pipelines and environments are available in the AzureML system registry, "sonata-test-reg"
registry_ml_client = MLClient(credential, subscription_id="80c77c76-74ba-4c8c-8229-4c3b2957990c",
        resource_group_name="sonata-test-rg",
        workspace_name="sonata-test-ws")
    
model_name = "bert-base-uncased"
version_list = list(registry_ml_client.models.list(model_name))
if len(version_list) == 0:
    print("Model not found in registry")
else:
    model_version = version_list[0].version
foundation_model = registry_ml_client.models.get(model_name, model_version)
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
# Download a small sample of the dataset into the ./book-corpus-dataset directory
# %run ./book-corpus-dataset/download-dataset.py --download_dir ./book-corpus-dataset

# load the ./book-corpus-dataset/train.jsonl file into a pandas dataframe and show the first 5 rows
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
train_df = pd.read_json("./book-corpus-dataset/train.jsonl", lines=True)
train_df.head()
# Get the right mask token from huggingface
import urllib.request, json

with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_name}") as url:
    data = json.load(url)
    mask_token = data["mask_token"]

# take the value of the "text" column, replace a random word with the mask token and save the result in the "masked_text" column
import random, os

train_df["masked_text"] = train_df["text"].apply(
    lambda x: x.replace(random.choice(x.split()), mask_token, 1)
)
# save the train_df dataframe to a jsonl file in the ./book-corpus-dataset folder with the masked_ prefix
train_df.to_json(
    os.path.join(".", "book-corpus-dataset", "masked_train.jsonl"),
    orient="records",
    lines=True,
)
train_df.head()
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)

# Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
timestamp = int(time.time())
online_endpoint_name = "fill-mask-" + str(timestamp)
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for "
    + foundation_model.name
    + ", for fill-mask task",
    auth_mode="key",
)
workspace_ml_client.begin_create_or_update(endpoint).wait()
# create a deployment
demo_deployment = ManagedOnlineDeployment(
    name="fillmask",
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_DS3_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=60000,
    ),
)
workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
endpoint.traffic = {"fillmask": 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
import json

# read the ./book-corpus-dataset/masked_train.jsonl file into a pandas dataframe
df = pd.read_json("./book-corpus-dataset/masked_train.jsonl", lines=True)
# escape single and double quotes in the masked_text column
df["masked_text"] = df["masked_text"].str.replace("'", "\\'").str.replace('"', '\\"')
# pick 1 random row
sample_df = df.sample(1)
# # create a json object with the key as "inputs" and value as a list of values from the masked_text column of the sample_df dataframe
# test_json = {"inputs": {"input_string": sample_df["masked_text"].tolist()}}
# # save the json object to a file named sample_score.json in the ./book-corpus-dataset folder
# with open(os.path.join(".", "book-corpus-dataset", "sample_score.json"), "w") as f:
#     json.dump(test_json, f)
# sample_df.head()
# # compare the predicted squences with the ground truth sequence
# compare_df = pd.DataFrame(
#     {
#         "ground_truth_sequence": sample_df["text"].tolist(),
#         "predicted_sequence": [
#             sample_df["masked_text"].tolist()[0].replace(mask_token, response_df[0][0])
#         ],
#     }
# )
# compare_df.head()
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
