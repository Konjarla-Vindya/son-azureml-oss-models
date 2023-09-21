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
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset
import numpy as np
import evaluate

def create_training_args(model_name, task, batch_size=16, num_train_epochs=3):
    return TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
    )

def create_data_collator(tokenizer):
    return DataCollatorForTokenClassification(tokenizer)

def create_compute_metrics(label_list):
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def tokenize_and_align_labels(dataset, tokenizer, task, label_all_tokens=True):
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align(examples),
        batched=True,
    )

    return tokenized_datasets




def load_custom_dataset(dataset_name, task, batch_size):
    # Load the dataset
    datasets = load_dataset(dataset_name)

    # Define label_list based on the task
    if task == "ner":
        label_list = datasets["train"].features[f"{task}_tags"].feature.names
    else:
        label_list = None

   
    print(f"Loaded dataset: {dataset_name}")
    print(f"Task: {task}")
    print(f"Label List: {label_list}")
    print(f"Batch Size: {batch_size}")


    return datasets, label_list, batch_size



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
  tokenizer=loaded_model.tokenizer
  dataset_name = "conll2003"
  task = "ner"
  batch_size = 16
  datasets, label_list, batch_size = load_custom_dataset(dataset_name, task, batch_size)
  #task = "ner"
  label_all_tokens = True
  tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens)
  model_name = test_model_name

  num_train_epochs = 3

  training_args = create_training_args(model_name, task, batch_size, num_train_epochs)
  data_collator = create_data_collator(tokenizer)
  label_list = label_list  # You should define label_list based on your dataset
  compute_metrics_fn = create_compute_metrics(label_list)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=subset_train_dataset,  
    eval_dataset=subset_validation_dataset,  
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_fn
   )

  fine_tune_results = trainer.train()
  print(fine_tune_results)
  evaluation_results = trainer.evaluate()
  print(evaluation_results)
  #data_set()
    
