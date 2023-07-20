from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
    AzureCliCredential,
)

from azure.ai.ml.entities import AmlCompute
import time
import json
import os

test_model_name = os.environ.get('test_model_name')
test_queue = os.environ.get('test_queue')
test_trigger_next_model = os.environ.get('test_trigger_next_model')
test_sku_type = os.environ.get('test_sku_type')
test_set = os.environ.get('test_set')
test_keep_looping = os.environ.get('test_keep_looping')








def get_sku_override():
    try:
        with open(f'../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print (f"::warning:: Could not find sku-override file: \n{e}")
        return None

def get_test_queue():
    config_name = test_queue+'-test'
    queue_file1 = f"../config/queue/{test_set}/{config_name}.json"
    queue_file = f"../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return json.load(f)

def set_next_trigger_model(queue):
    print ("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    index = queue['models'].index(test_model_name)
    print (f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(queue['models']) - 1:
        next_model = queue['models'][index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            print ("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def create_online_endpoint():
# def Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
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

    
def test_deploy(online_endpoint_name):

      
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
    



def delete_endpoint(online_endpoint_name):
     
    workspace_ml_client = MLClient.online_endpoints.begin_delete(name=online_endpoint_name).wait()

queue = get_test_queue() 


def data_set():

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

credential = DefaultAzureCredential()
registry_ml_client = MLClient(credential, registry_name="sonata-test-reg")
 
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

def main():

    # constants
    check_override = True

    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        print ("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit (1)

    

    sku_override = get_sku_override()
    if sku_override is None:
        check_override = False

    if test_trigger_next_model == "true":
        set_next_trigger_model(queue)

    # print values of all above variables
    print (f"test_subscription_id: {queue['subscription']}")
    print (f"test_resource_group: {queue['subscription']}")
    print (f"test_workspace_name: {queue['workspace']}")
    print (f"test_model_name: {test_model_name}")
    print (f"test_sku_type: {test_sku_type}")
    print (f"test_registry: queue['registry']")
    print (f"test_trigger_next_model: {test_trigger_next_model}")
    print (f"test_queue: {test_queue}")
    print (f"test_set: {test_set}")


    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print ("::error:: Auth failed, DefaultAzureCredential not working: \n{e}")
        exit (1)

    # connect to workspace
    workspace_ml_client = MLClient(
        credential=credential, 
        subscription_id=queue['subscription'],
        resource_group_name=queue['resource_group'],
        workspace_name=queue['workspace']

    )

    registry_ml_client = MLClient(
        credential=credential, 
        registry_name=queue['registry']
    )

 
    data_set()
    create_online_endpoint()
    test_deploy()
    delete_endpoint()
# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"


 

import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)

if __name__ == "__main__":
    main()

