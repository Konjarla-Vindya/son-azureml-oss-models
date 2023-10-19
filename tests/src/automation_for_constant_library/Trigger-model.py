import os
import json

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
for json_file in json_files:
    with open(os.path.join(json_directory, json_file), 'r') as file:
        data = json.load(file)
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            print(first_item)
            #trigger_model(first_item)
if __name__ == "__main__":
    first_item = data[0]
    print(first_item)
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
#                 trigger_model(first_item)

