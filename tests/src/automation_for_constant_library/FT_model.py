import os
from FT_initial_automation import load_model  
import mlflow
import transformers
import os
import torch
import json
import pandas as pd
import transformers
import mlflow
import datetime
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    ModelConfiguration,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer
)
from azureml.core import Workspace
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
# import evaluate
import argparse
import os
from azureml.core import Workspace
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
import numpy as np
import evaluate

import os
import json
import pandas as pd
import subprocess

# Define a function to download and prepare a dataset using a provided Python script
def download_and_prepare_dataset(dataset_name, download_script, download_dir1, label_mapping=None):
    # Create the download directory if it does not exist
    if not os.path.exists(download_dir1):
        os.makedirs(download_dir1)

    # Run the provided download script to download the dataset
    exit_status = subprocess.call(f"python {download_script} --download_dir1 {download_dir1}", shell=True)
    if exit_status != 0:
        raise Exception(f"Error downloading {dataset_name} dataset")

    # Load the JSONL file into a pandas DataFrame and show the first 5 rows
    pd.set_option("display.max_colwidth", 0)
    df = pd.read_json(os.path.join(download_dir1, "train.jsonl"), lines=True)
    df.head()

    # Load the id2label JSON element of the label.json file into a pandas table
    with open(os.path.join(download_dir1, "label.json")) as f:
        id2label = json.load(f)["id2label"]
        label_df = pd.DataFrame.from_dict(id2label, orient="index", columns=["label_string"])
        label_df["label"] = label_df.index.astype("int64")
        label_df = label_df[["label", "label_string"]]
        label_df.head()

    print(f"{dataset_name} dataset downloaded and prepared successfully in {download_dir1} -------------")

if __name__ == "__main__":
    # Download and prepare the SQuAD dataset using the provided script
    download_and_prepare_dataset("SQuAD", "./download-dataset-squad.py", "squad-dataset", label_mapping=None)


# def get_library_to_load_model(self, task: str) -> str:
#         """ Takes the task name and load the  json file findout the library 
#         which is applicable for that task and retyrun it 

#         Args:
#             task (str): required the task name 
#         Returns:
#             str: return the library name
#         """
#         try:
#             with open(FILE_NAME) as f:
#                 model_with_library = ConfigBox(json.load(f))
#                 print(f"scoring_input file:\n\n {model_with_library}\n\n")
#         except Exception as e:
#             print(
#                 f"::warning:: Could not find scoring_file: {model_with_library}. Finishing without sample scoring: \n{e}")
#         return model_with_library.get(task)

# def download_model_and_tokenizer(self, task: str) -> dict:
#         model_library_name = self.get_library_to_load_model(task=task)
#         print("Library name is this one : ", model_library_name)
#         # Load the library from the transformer
#         model_library = getattr(transformers, model_library_name)
#         # From the library load the model
#         model_name = loaded_model.model 
#         model_name.config.num_labels = 6 
#         model_name.classifier = torch.nn.Linear(model_name.config.hidden_size, model_name.config.num_labels) 
#         Text_classification_model = model_library.from_pretrained(
#             config=model_name.config) 
#         Text_classification_model.load_state_dict(model_name.state_dict(), strict=False)
     
if __name__ == "__main__":
  model_source_uri=os.environ.get('model_source_uri')
  test_model_name = os.environ.get('test_model_name')
  print("test_model_name-----------------",test_model_name)
  loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
  print("loaded_model---------------------",loaded_model)
  data_set()
