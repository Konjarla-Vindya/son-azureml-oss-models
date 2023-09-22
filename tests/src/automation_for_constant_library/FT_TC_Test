import os
import transformers
import torch
import json
import pandas as pd
import transformers
import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np

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
        label_all_tokens=True,
        num_labels=len(label_list)
    )

def fine_tune_token_classification(model_name, task):
    loaded_model = transformers.pipeline(task="text-classification", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_name = "conll2003"
    batch_size = 16
    datasets = load_dataset(dataset_name)
    label_list = datasets["train"].features[f"{task}_tags"].feature.names
    tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens=True)
    training_args = create_training_args(model_name, task, batch_size=batch_size, num_train_epochs=3)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics_fn = load_metric("seqeval")
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    subset_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    subset_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))
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
    evaluation_results = trainer.evaluate()
    save_directory = f"./fine_tuned_model_tokenclassification_{model_name}_{task}"
    trainer.save_model(save_directory)
    fine_tuned_model = AutoModelForTokenClassification.from_pretrained(save_directory)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model_pipeline = transformers.pipeline(task="token-classification", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    return model_pipeline

if __name__ == "__main__":
  model_name = os.environ.get('test_model_name')
  task = "ner"
  model_pipeline = fine_tune_token_classification(model_name, task)
