from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
import json
import os
import argparse
import sys
from util import load_model_list_file, get_model_containers

# constants
LOG = True

# parse command line argument to specify the directory to write the workflow files to
parser = argparse.ArgumentParser()
# mode - options are file or registry
parser.add_argument("--mode", type=str, default="file")
# registry name if model is in registry
parser.add_argument("--registry_name", type=str, default="HuggingFace")
# argument to specify Github workflow directory. can write to local dir for testing
# !!! main workflow files will be overwritten if set to "../../.github/workflows" !!!
parser.add_argument("--workflow_dir", type=str, default="../../.github/workflows")
# argument to specify queue directory
parser.add_argument("--queue_dir", type=str, default="../config/queue")
# queue set name (will create a folder under queue_dir with this name)
# !!! backup files in this folder will be overwritten !!!
parser.add_argument("--test_set", type=str, default="huggingface-all")
# file containing list of models to test, one per line
parser.add_argument("--model_list_file", type=str, default="../config/modellist.txt")
# test_keep_looping, to keep looping through the queue after all models have been tested
parser.add_argument("--test_keep_looping", type=str, default="false")
# test_trigger_next_model, to trigger next model in queue after each model is tested
parser.add_argument("--test_trigger_next_model", type=str, default="true")
# test_sku_type, to specify sku type to use for testing
parser.add_argument("--test_sku_type", type=str, default="cpu")
# parallel_tests, to specify number of parallel tests to run per workspace. 
# this will be used to create multiple queues
parser.add_argument("--parallel_tests", type=int, default=3)
# workflow-template.yml file to use as template for generating workflow files
parser.add_argument("--workflow_template", type=str, default="../config/workflow-template-huggingface.yml")
# workspace_list file get workspace metadata
parser.add_argument("--workspace_list", type=str, default="../config/workspaces.json")
# directory to write logs
parser.add_argument("--log_dir", type=str, default="../logs")

args = parser.parse_args()
parallel_tests = int(args.parallel_tests)

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print ("::error Auth failed, DefaultAzureCredential not working: \n{e}")
    exit (1)
# Connect to the HuggingFaceHub registry
registry_ml_client = MLClient(credential, registry_name="HuggingFace")

queue = []

# move this to config file later
templates=['transformers-cpu-small', 'transformers-cpu-medium', 'transformers-cpu-large','transformers-cpu-extra-large', 'transformers-gpu-medium']


def load_workspace_config():
    with open(args.workspace_list) as f:
        return json.load(f)
    
# function to assign models to queues
# assign each model from models to a thread per workspace in a round robin fashion by appending to a list called 'models' in the queue dictionary
def assign_models_to_queues(models, workspace_list):
  queue = {}
  i=0
  while i < len(models):
      for workspace in workspace_list:
          print (f"workspace instance: {workspace}")
          for thread in range(parallel_tests):
              print (f"thread instance: {thread}")
              if i < len(models):
                  if workspace not in queue:
                      queue[workspace] = {}
                      print("queue[workspace]",queue[workspace])
                  if thread not in queue[workspace]:
                      queue[workspace][thread] = []
                  queue[workspace][thread].append(models[i])
                  print("queue[workspace][thread]",queue[workspace][thread])
                  i=i+1
def main():
    # get list of models from registry
    if args.mode == "registry":
        models = get_model_containers(args.registry_name)
    elif args.mode == "file":
        models = load_model_list_file(args.model_list_file)
    else:
        print (f"::error Invalid mode {args.mode}")
        exit (1)
    print (f"Found {len(models)} models")
    # load workspace_list_json
    workspace_list = load_workspace_config()
    print (f"Found {len(workspace_list)} workspaces")
    # assign models to queues
    queue = assign_models_to_queues(models, workspace_list)
    print (f"Created queues")
    # create queue files
    create_queue_files(queue, workspace_list)
    print (f"Created queue files")
    # create workflow files
    create_workflow_files(queue, workspace_list)
    print (f"Created workflow files")
    print (f"Summary:")
    print (f"  Models: {len(models)}")
    print (f"  Workspaces: {len(workspace_list)}")
    print (f"  Parallel tests: {parallel_tests}")
    print (f"  Total queues: {len(workspace_list)*parallel_tests}")
    print (f"  Average models per queue: {int(len(models)/(len(workspace_list)*parallel_tests))}")

        
if __name__ == "__main__":
    main()
