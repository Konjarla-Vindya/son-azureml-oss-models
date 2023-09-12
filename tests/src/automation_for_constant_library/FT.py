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
# import evaluate
import pandas as pd


subscription_id = '80c77c76-74ba-4c8c-8229-4c3b2957990c'
resource_group = 'sonata-test-rg'
workspace_name = 'sonata-test-ws'
credential = DefaultAzureCredential()
ws = Workspace(subscription_id, resource_group, workspace_name)

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
ml_client = MLClient(credential, subscription_id, resource_group, ws)
workspace_ml_client = MLClient(
    credential,
    subscription_id="80c77c76-74ba-4c8c-8229-4c3b2957990c",
    resource_group_name="sonata-test-rg",
    workspace_name="sonata-test-ws",
)

# download the dataset using the helper script. This needs datasets library: https://pypi.org/project/datasets/
import os

test_model_name = os.environ.get('test_model_name')
# bert-base-uncased

version_list = list(workspace_ml_client.models.list(test_model_name))
foundation_model = ''
if len(version_list) == 0:
    print("Model not found in registry")
else:
    model_version = version_list[0].version
    foundation_model = workspace_ml_client.models.get(test_model_name, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for F.T".format(
            foundation_model.name, foundation_model.version, foundation_model.id
        )
    )
print (f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")

model_source_uri = foundation_model.properties["mlflow.modelSourceUri"]
loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri)

# download the dataset using the helper script. This needs datasets library: https://pypi.org/project/datasets/
import os

exit_status = os.system("python ./download-dataset.py --download_dir emotion-dataset")
if exit_status != 0:
    raise Exception("Error downloading dataset")
# load the ./emotion-dataset/train.jsonl file into a pandas dataframe and show the first 5 rows
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
df = pd.read_json("./emotion-dataset/train.jsonl", lines=True)
df.head()

# load the id2label json element of the ./emotion-dataset/label.json file into pandas table with keys as 'label' column of int64 type and values as 'label_string' column as string type
import json

with open("./emotion-dataset/label.json") as f:
    id2label = json.load(f)
    id2label = id2label["id2label"]
    label_df = pd.DataFrame.from_dict(
        id2label, orient="index", columns=["label_string"]
    )
    label_df["label"] = label_df.index.astype("int64")
    label_df = label_df[["label", "label_string"]]
label_df.head()

import torch

from transformers import BertConfig, BertForSequenceClassification

bert_model = loaded_model.model 

bert_model.config.num_labels = 6 

bert_model.classifier = torch.nn.Linear(bert_model.config.hidden_size, bert_model.config.num_labels) 

Text_classification_model = BertForSequenceClassification(

    config=bert_model.config

) 

Text_classification_model.load_state_dict(bert_model.state_dict(), strict=False)

import pandas as pd


test_df = pd.read_json("./emotion-dataset/test.jsonl", lines=True).head(512)
train_df = pd.read_json("./emotion-dataset/train.jsonl", lines=True).head(512)
validation_df = pd.read_json("./emotion-dataset/validation.jsonl", lines=True).head(512)


train_df = train_df.merge(label_df, on="label", how="left")
validation_df = validation_df.merge(label_df, on="label", how="left")
test_df = test_df.merge(label_df, on="label", how="left")

print(train_df.head())

# save 10% of the rows from the train, validation and test dataframes into files with small_ prefix in the ./emotion-dataset folder
frac = 1
train_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_test.jsonl", orient="records", lines=True
)

tokenizer=loaded_model.tokenizer

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
from datasets import Dataset


train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(test_df)


def tokenize_function(examples):
    tokenized_output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    tokenized_output["labels"] = examples["label"]
    return tokenized_output

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=False)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=False)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5)


trainer = Trainer(
        model=Text_classification_model,
        args=training_args,
        
        #data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset
)

result = trainer.train()

save_directory = "./test_trainer"

trainer.save_model(save_directory)

from transformers import AutoModelForSequenceClassification


fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)

model_pipeline = transformers.pipeline(task="text-classification", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer )
timestamp_uuid = datetime.datetime.now().strftime("%m%d%H%M%f")
model_name = f"FT-TC-{test_model_name}"
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=model_pipeline,
        artifact_path=model_name
    )
registered_model = mlflow.register_model(model_info.model_uri, model_name)

import mlflow

version_list = list(workspace_ml_client.models.list(registered_model.name))

foundation_model = ''

if len(version_list) == 0:

    print("Model not found in registry")

else:

    model_version = version_list[0].version

    foundation_model = workspace_ml_client.models.get(registered_model.name, model_version)

    print(

        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(

            foundation_model.name, foundation_model.version, foundation_model.id

        )

    )

print (f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")

 

model_sourceuri = foundation_model.properties["mlflow.modelSourceUri"]

model_pipeline1 = mlflow.transformers.load_model(model_uri=model_sourceuri)

from box import ConfigBox

text_classification = ConfigBox(
    {

    "inputs": [

      "Im good",

      "I am bad"

    ]
}
)
model_pipeline1(text_classification.inputs)







