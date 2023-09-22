import os
import mlflow
import transformers
import numpy as np
from datasets import load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

def fine_tune_model(model_name, task):
    # Define functions
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

        tokenized_datasets = dataset.map(
            lambda examples: tokenize_and_align(examples),
            batched=True,
        )

        return tokenized_datasets

    def load_custom_dataset(dataset_name, task, batch_size):
        datasets = load_dataset(dataset_name)

        if task == "ner":
            label_list = datasets["train"].features[f"{task}_tags"].feature.names
        else:
            label_list = None

        print(f"Loaded dataset: {dataset_name}")
        print(f"Task: {task}")
        print(f"Label List: {label_list}")
        print(f"Batch Size: {batch_size}")

        return datasets, label_list, batch_size

    # Load model and tokenizer
    model_source_uri = os.environ.get('model_source_uri')
    loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
    tokenizer = loaded_model.tokenizer

    # Load dataset and preprocess
    dataset_name = "conll2003"
    batch_size = 16
    datasets, label_list, batch_size = load_custom_dataset(dataset_name, task, batch_size)
    label_all_tokens = True
    tokenized_datasets = tokenize_and_align_labels(datasets, tokenizer, task, label_all_tokens)
    metric = load_metric("seqeval")
    
    # Initialize training arguments, model, and trainer
    num_train_epochs = 3
    training_args = create_training_args(model_name, task, batch_size, num_train_epochs)
    data_collator = create_data_collator(tokenizer)
    compute_metrics_fn = create_compute_metrics(label_list)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
    
    # Subset and fine-tune
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
    print(fine_tune_results)
    evaluation_results = trainer.evaluate()
    print(evaluation_results)
    
    # Save the fine-tuned model
    save_directory = f"./fine_tuned_model_{task}"
    trainer.save_model(save_directory)
    fine_tuned_model = AutoModelForTokenClassification.from_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print(fine_tuned_model)
    print(tokenizer)
    mlflow.end_run()
    
    # Register the model with MLflow
    model_pipeline = transformers.pipeline(task="token-classification", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
    model_name = f"ft-tkc-{model_name}"
    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=model_pipeline,
            artifact_path=model_name
        )
    
    registered_model = mlflow.register_model(model_info.model_uri, model_name)
    print(registered_model)

if __name__ == "__main__":
    model_name = os.environ.get('test_model_name')
    task = "ner"
    fine_tune_model(model_name, task)
