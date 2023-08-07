import subprocess

# List of Hugging Face model names to deploy
model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    # Add more model names as needed
]

# Trigger the GitHub Actions workflow for each model
for model_name in model_names:
    command = f"gh workflow run bert-base-uncased --ref main --inputs model_name={model_name}"
    subprocess.run(command, shell=True, check=True)
