import os
import mlflow
import transformers
import json
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    Model,
    ModelPackage,
    Environment,
    CodeConfiguration,
    AzureMLOnlineInferencingServer
)
from azureml.core import Workspace
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_metric

# Read configuration from JSON file
config_file = "./token_config.json"#dataset_task.json
#config_file = "./dataset_task.json"
with open(config_file, "r") as json_file:
    config = json.load(json_file)

dataset_name = config["dataset_name"]
batch_size = config["batch_size"]
num_train_epochs = config["num_train_epochs"]
max_length = config["max_length"]
doc_stride = config["doc_stride"]
task = config["task"]
num_labels = config.get("num_labels", None)  # Use default value of None if not specified

def create_training_args(model_name, task, batch_size=batch_size, num_train_epochs=num_train_epochs):
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


def tokenize_and_prepare_features(dataset, tokenizer, task, max_length=max_length, doc_stride=doc_stride):
    def prepare_train_features(examples):
        # answers = examples.get("answers", {"answer_start": [], "text": []})

        tokenized_examples = tokenizer(
            examples["question" if tokenizer.padding_side == "right" else "context"],
            examples["context" if tokenizer.padding_side == "right" else "question"],
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]

            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])


                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if tokenizer.padding_side == "right" else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
        


    if task == "qa":
        return dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    elif task == "ner":
    
        return dataset.map(
            lambda examples: tokenize_and_align(examples),
            batched=True,
           
        )
    else:
        raise ValueError("Unsupported task: " + task)

def tokenize_and_align_labels(dataset, tokenizer, task, label_all_tokens=True):
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    if task == "qa":
        return dataset.map(
            prepare_train_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    elif task == "ner":
    
        return dataset.map(
            lambda examples: tokenize_and_align(examples),
            batched=True,
           
        )
    else:
        raise ValueError("Unsupported task: " + task)

def load_custom_dataset(dataset_name, task, batch_size):
    datasets = load_dataset(dataset_name)

    if task == "ner":
        label_list = datasets["train"].features[f"{task}_tags"].feature.names
        batch_size = config["batch_size"]
    else:
        label_list = None

    print(f"Loaded dataset: {dataset_name}")
    print(f"Task: {task}")
    print(f"Label List: {label_list}")
    print(f"Batch Size: {batch_size}")

    return datasets, label_list, batch_size
def create_compute_metrics(label_list):
        metric = load_metric("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

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


def fine_tune_model(model_name, task):
    # Load model and tokenizer
    model_source_uri = os.environ.get('model_source_uri')
    loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
    tokenizer = loaded_model.tokenizer
    batch_size = config["batch_size"]

    # Define TrainingArguments
    training_args = create_training_args(model_name, task, batch_size=batch_size, num_train_epochs=num_train_epochs)

    datasets, label_list, batch_size = load_custom_dataset(dataset_name, task, batch_size)


    if task == "qa":
        tokenized_datasets = tokenize_and_prepare_features(datasets, tokenizer, task=task)
    else:
        tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens=True)
        metric = load_metric("seqeval")
        
        
    num_labels = None if task == "qa" else 9  # Adjust as needed
    model = (
        AutoModelForQuestionAnswering.from_pretrained(model_name)
        if task == "qa"
        else AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    )

    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

    if task == "qa":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=subset_validation_dataset,
            tokenizer=tokenizer,
        )
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer)
        compute_metrics_fn = create_compute_metrics(label_list)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=subset_train_dataset,
            eval_dataset=subset_validation_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )


    fine_tune_results = trainer.train()
    print(fine_tune_results)

    evaluation_results = trainer.evaluate()
    print(evaluation_results)

    save_directory = f"./fine_tuned_model_{task}"
    trainer.save_model(save_directory)
    fine_tuned_model = (
        AutoModelForQuestionAnswering.from_pretrained(save_directory)
        if task == "qa"
        else AutoModelForTokenClassification.from_pretrained(save_directory)
    )
    tokenizer.save_pretrained(save_directory)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print(fine_tuned_model)
    print(tokenizer)
    
    mlflow.end_run()

  




    model_pipeline = (
        transformers.pipeline(task="question-answering", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
        if task == "qa"
        else transformers.pipeline(task="token-classification", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    )
    model_name = f"ft-{task}-{model_name}"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path=model_name
        )

    registered_model = mlflow.register_model(model_info.model_uri, model_name)
    print(registered_model)
    print("Registered Model : ",
              client.get_registered_model(registered_model))
        registered_model_detail = client.get_latest_versions(
            name=registered_model, stages=["None"])
        model_detail = registered_model_detail[0]
        print("Latest registered model version is : ", model_detail.version)
        loaded_model_pipeline = mlflow.transformers.load_model(
            model_uri=model_detail.source, return_type="pipeline")
        from box import ConfigBox
        token_classification = ConfigBox(
        {
        "inputs": [
        "My name is Amarah",
        "I am from Jamshedpur"
        ]
        }
        )
        loaded_model_pipeline(token_classification.inputs)
 





if __name__ == "__main__":
    model_name = os.environ.get('test_model_name')
    fine_tune_model(model_name, task)
   
