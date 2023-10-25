import os
import json

# Directory where your JSON files are located
json_directory = "../../config/queue/huggingface-all"  

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

# Sort the list of JSON files to process them in a specific order
json_files.sort()

if len(json_files) < 2:
    print("There are not enough JSON files to proceed.")
else:
    # Open the first JSON file
    with open(os.path.join(json_directory, json_files[0]), 'r') as file1:
        data1 = json.load(file1)
        strings1 = data1.get("strings", [])

    # Open the second JSON file
    with open(os.path.join(json_directory, json_files[1]), 'r') as file2:
        data2 = json.load(file2)
        strings2 = data2.get("strings", [])

    # Determine the maximum number of strings in both files
    max_strings = max(len(strings1), len(strings2))

    for i in range(max_strings):
        if i < len(strings1):
            print(f"From JSON 1: {strings1[i]}")

        if i < len(strings2):
            print(f"From JSON 2: {strings2[i]}")
