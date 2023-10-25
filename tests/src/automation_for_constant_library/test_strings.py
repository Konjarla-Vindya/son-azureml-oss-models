import os
import json
import requests
import subprocess

# Directory where your JSON files are located
json_directory = "../../config/queue/huggingface-all"  #"/path/to/json/files"

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# Sort the list of JSON files to process them in a specific order
json_files.sort()

# Function to trigger the model on JSON data
def trigger_model(data):
    # Replace this with code to trigger your model
    print("Model triggered with data:", data)

# Iterate through JSON files
for json_file in json_files[:2]:
    with open(os.path.join(json_directory, json_file), 'r') as file:
        data = json.load(file)
        data1=data["models"][0]
        print(data["models"][0])
        repository_owner = "Konjarla-Vindya"
        repository_name = "son-azureml-oss-models"
        github_token = "WORKFLOW_TOKEN"
        file_path = "data1.yml"  # Replace with the desired YAML file path
        api_url = "https://github.com/Konjarla-Vindya/son-azureml-oss-models/blob/main/.github/workflows/HELLO_WORLD.yml" 
        print(api_url)
        headers = {
           "Authorization": f"token {github_token}",
          "Accept": "application/vnd.github.v3+json"
        }
        payload = {
             "ref": "main"
        }
        response = requests.post(api_url, headers=headers,json=payload)
        
        if response.status_code == 204:
            print(f"Workflow '{data1}' has been triggered successfully.")
        else:
            print(f"Failed to trigger the workflow. Status code: {response.status_code}")
            print(response.text)
