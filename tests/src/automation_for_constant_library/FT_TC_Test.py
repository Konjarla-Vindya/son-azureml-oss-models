import os
import mlflow
import transformers
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)

def fine_tune_and_register_model(model_name, task):
    # Define the dataset, batch size, and other relevant parameters
    dataset_name = "conll2003"
    batch_size = 16
    num_train_epochs = 3
    label_all_tokens = True

    # Load the dataset
    datasets = load_dataset(dataset_name)
    label_list = datasets["train"].features[f"{task}_tags"].feature.names if task == "ner" else None

    # Load the tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens)

    # Define training arguments and data collator
    training_args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Define the model for token classification
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
    
    # Create a subset of the training and validation datasets
    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))
    
    # Create a Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=subset_train_dataset,
        eval_dataset=subset_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=create_compute_metrics(label_list),
    )

    # Fine-tune the model
    fine_tune_results = trainer.train()
    evaluation_results = trainer.evaluate()

    # Save the fine-tuned model and tokenizer
    save_directory = f"./fine_tuned_model_{task}"
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # Create a pipeline for inference with the fine-tuned model
    model_pipeline = pipeline(task="token-classification", model=save_directory, tokenizer=save_directory)

    # Register the fine-tuned model with MLflow
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path=f"ft-tkc-{model_name}-{task}"
        )
    
    registered_model = mlflow.register_model(model_info.model_uri, f"ft-tkc-{model_name}-{task}")
    
    return registered_model

if __name__ == "__main__":
    print("Model Name:", model_name)
    print("Task:", task)
    registered_model = fine_tune_and_register_model(model_name, task)
    print("Registered Model:", registered_model)

if __name__ == "__main__":
  model_name = os.environ.get('test_model_name')
  task = "ner"
  print("Model Name:", model_name)
  print("Task:", task)
  registered_model = fine_tune_and_register_model(model_name, task)
  print("Registered Model:", registered_model)
