import argparse
import os
import json
from datasets import load_dataset, load_metric

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--download_dir1", type=str, default="squad-dataset", help="Directory to download the SQuAD dataset")
args = parser.parse_args()

# Create the download directory if it does not exist
if not os.path.exists(args.download_dir1):
    os.makedirs(args.download_dir1)

# Load the SQuAD dataset
squad_dataset = load_dataset("squad")

# Save the train and validation splits as JSONL files
squad_dataset["train"].to_json(os.path.join(args.download_dir1, "train.jsonl"), orient="records", lines=True)
squad_dataset["validation"].to_json(os.path.join(args.download_dir1, "validation.jsonl"), orient="records", lines=True)

# Create a label mapping for SQuAD (questions and answers)
label_mapping = {
    "id2label": {0: "no-answer", 1: "yes", 2: "no"},
    "label2id": {"no-answer": 0, "yes": 1, "no": 2}
}

# Save the label mapping as label.json
with open(os.path.join(args.download_dir1, "label.json"), "w") as label_file:
    json.dump(label_mapping, label_file)

print("SQuAD dataset downloaded and prepared successfully.")
