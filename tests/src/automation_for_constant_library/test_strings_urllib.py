import os
import json
import urllib.request

# Directory where your JSON files are located
json_directory = "../../config/queue/huggingface-all"

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# Sort the list of JSON files to process them in a specific order
json_files.sort()

# Function to trigger the model on JSON data
def trigger_model(data):
  # Replace this with code to trigger your model
  print("Model triggered with data:", data)

# repository_owner = "Konjarla-Vindya"
# repository_name = "son-azureml-oss-models"
# github_token = "WORKFLOW_TOKEN"
# workflow_file_path = "data1.yml"  # Replace with the desired YAML file path
# api_url = f"https://api.github.com/repos/{repository_owner}/{repository_name}/actions/workflows/{data1}.yml/dispatches"

# Iterate through JSON files
for json_file in json_files[:2]:
  with open(os.path.join(json_directory, json_file), 'r') as file:
    data = json.load(file)
    data1 = data["models"][0]
    print(data1)
    repository_owner = "Konjarla-Vindya"
    repository_name = "son-azureml-oss-models"
    github_token = "WORKFLOW_TOKEN"
    workflow_file_path = "data1.yml"  # Replace with the desired YAML file path
    print(workflow_file_path)
    api_url = f"https://api.github.com/repos/{repository_owner}/{repository_name}/actions/workflows/{data1}.yml/dispatches"
    print(api_url)

    # Trigger the GitHub Action workflow using the urllib library
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main"  # Replace with the branch where your workflow is located
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(api_url, data, headers)
    response = urllib.request.urlopen(req)

    if response.getcode() == 204:
        print(f"Workflow '{workflow_file_path}' has been triggered successfully.")
    else:
        print(f"Failed to trigger the workflow. Status code: {response.getcode()}")
