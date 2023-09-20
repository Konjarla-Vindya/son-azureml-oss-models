import os
import time
import json
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Model
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline,
    AutoConfig,
)

def load_azure_credentials(credentials):
    # Load Azure credentials from a JSON file
    with open(credentials, "r") as creds_file:
        azure_credentials = json.load(creds_file)
    return azure_credentials

def authenticate_to_azure(azure_credentials):
    # Authenticate to Azure Machine Learning workspace
    subscription_id = azure_credentials["subscription_id"]
    resource_group = azure_credentials["resource_group"]
    workspace_name = azure_credentials["workspace_name"]
    
    credential = DefaultAzureCredential()
    ws = Workspace(subscription_id, resource_group, workspace_name)
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    return ws

def download_model(model_name, model_version, target_dir):
    # Download the model to a local directory
    model = Model(ws, name=model_name, version=model_version)
    model.download(target_dir=target_dir, exist_ok=True)

def prepare_and_fine_tune_model(tokenizer, model, datasets):
    max_length = 384  # The maximum length of a feature (question and context)
    doc_stride = 128  # The authorized overlap between two parts of the context when splitting it is needed.
    pad_on_right = tokenizer.padding_side == "right"

    
def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
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

        # If no answers are given, set the cls_index as the answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise, move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

    # Load and preprocess the dataset
    tokenized_datasets = datasets.map(
        prepare_train_features, batched=True, remove_columns=datasets["train"].column_names
    )

    # Fine-tuning the model
    batch_size = 16
    args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch")
    data_collator = default_data_collator

    subset_train_indices = range(100)
    subset_validation_indices = range(100)

    subset_train_dataset = tokenized_datasets["train"].select(subset_train_indices)
    subset_validation_dataset = tokenized_datasets["validation"].select(subset_validation_indices)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=subset_train_dataset,
        eval_dataset=subset_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    fine_tune_results = trainer.train()
    evaluation_results = trainer.evaluate()

    # Save the fine-tuned model
    model_save_directory = "./ft_qa_model"
    trainer.save_model(model_save_directory)
    ft_model = AutoModelForQuestionAnswering.from_pretrained(model_save_directory)
    tokenizer.save_pretrained(model_save_directory)
    ft_tokenizer = AutoTokenizer.from_pretrained(model_save_directory)

    return ft_model, ft_tokenizer

def register_and_deploy_model(ft_model, ft_tokenizer, model_name):
    model_pipeline = pipeline(task="question-answering", model=ft_model, tokenizer=ft_tokenizer)
    
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline, artifact_path=model_name
        )
    registered_model = mlflow.register_model(model_info.model_uri, model_name)

    timestamp = int(time.time())
    online_endpoint_name = "qa-" + str(timestamp)

    # Create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="Online endpoint for " + registered_model.name + ", qa",
        auth_mode="key",
    )
    workspace_ml_client.begin_create_or_update(endpoint).wait()

    # Create a deployment
    demo_deployment = ManagedOnlineDeployment(
        name="demo",
        endpoint_name=online_endpoint_name,
        model=registered_model.id,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        liveness_probe=ProbeSettings(initial_delay=600),
    )
    workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    endpoint.traffic = {"demo": 100}
    workspace_ml_client.begin_create_or_update(endpoint).result()

    print(f"Model deployed as an online endpoint: {online_endpoint_name}")

if __name__ == "__main__":
    # Load Azure credentials from a JSON file
    azure_credentials = load_azure_credentials("azure_credentials.json")
    
    # Authenticate to Azure Machine Learning workspace
    ws = authenticate_to_azure(azure_credentials)
    
    # Load model details from a JSON file
    model_details = load_model_details("model_details.json")
    model_name = model_details["model_name"]
    model_version = model_details["model_version"]
    
    # Download the model
    download_model(model_name, model_version, "./downloaded_models")
    
    # Load the downloaded tokenizer and model
    model_component_path = "./downloaded_models/{}/components/tokenizer".format(model_name)
    model_model_path = "./downloaded_models/{}/model".format(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_component_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_model_path)
    
    # Load the SQuAD dataset
    from datasets import load_dataset
    datasets = load_dataset("squad")
    
    # Prepare, fine-tune, register, and deploy the model
    ft_model, ft_tokenizer = prepare_and_fine_tune_model(tokenizer, model, datasets)
    register_and_deploy_model(ft_model, ft_tokenizer, model_name)
