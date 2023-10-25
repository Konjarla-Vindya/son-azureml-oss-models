import os
import json

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

# Iterate through JSON files
for json_file in json_files[:2]:
    with open(os.path.join(json_directory, json_file), 'r') as file:
        data = json.load(file)
        data1=data["models"][0]
        print(data["models"][0])
