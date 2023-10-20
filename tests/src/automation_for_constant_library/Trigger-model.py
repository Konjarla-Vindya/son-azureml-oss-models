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
        # if isinstance(data, list) and len(data) > 0:
        #     first_item = data[0]
        #     print(first_item)
         repository_owner = "Konjarla-Vindya"
         repository_name = "son-azureml-oss-models"
         file_path = ".github/workflows/data1.yml"  # Replace with the desired YAML file path
         api_url = f"https://api.github.com/repos/{repository_owner}/{repository_name}/blob/main/{file_path}"
         print(api_url)
        #     subprocess.run(['gh', 'workflow', 'run', api_url])




    
    





























# import os
# import json

# # Directory where your JSON files are located
# json_directory = "../../config/queue/huggingface-all"  #"/path/to/json/files"

# # List all JSON files in the directory
# json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# # Sort the list of JSON files to process them in a specific order
# json_files.sort()

# # Function to trigger the model on JSON data
# def trigger_model(data):
#     # Replace this with code to trigger your model
#     print("Model triggered with data:", data)

# if __name__ == "__main__":
#     for json_file in json_files:
#         with open(os.path.join(json_directory, json_file), 'r') as file:
#             data = json.load(file)
#             if isinstance(data, list) and len(data) > 0:
#                 first_item = data[0]
#                 print(first_item)
#                 #trigger_model(first_item)
# import os
# import json

# # Directory where your JSON files are located
# json_directory = "../../config/queue/huggingface-all"  #"/path/to/json/files"

# if not os.path.exists(json_directory):
#     print(f"Directory '{json_directory}' does not exist.")
#     exit(1)

# # List all JSON files in the directory
# json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# # Sort the list of JSON files to process them in a specific order
# json_files.sort()

# # Function to trigger the model on JSON data
# def trigger_model(data):
#     # Replace this with code to trigger your model
#     print("Model triggered with data:", data)

# # This block of code will only execute if this script is run directly, not when it's imported as a module.
# if __name__ == "__main__":
#     # Iterate through JSON files
#     for json_file in json_files:
#         file_path = os.path.join(json_directory, json_file)
#         try:
#             with open(file_path, 'r') as file:
#                 data = json.load(file)
#                 if isinstance(data, list) and len(data) > 0:
#                     first_item = data[0]
#                     print(first_item)
#                     # Uncomment the line below to trigger your model with the data.
#                     # trigger_model(first_item)
#         except Exception as e:
#             print(f"Error processing file '{file_path}': {str(e)}")

