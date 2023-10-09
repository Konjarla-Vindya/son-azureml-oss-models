import requests
import base64
import os

# Replace with your GitHub repository and PAT
repository = "Konjarla-Vindya/son-azureml-oss-models"
token = os.environ['token']

# Get the list of workflows for the repository
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}
response = requests.get(f"https://api.github.com/repos/{repository}/actions/workflows", headers=headers)
response.raise_for_status()

# Loop through each workflow
for workflow in response.json()["workflows"]:
    workflow_name = workflow["name"]
    workflow_id = workflow["id"]

    # Get the latest workflow run for the workflow
    response = requests.get(f"https://api.github.com/repos/{repository}/actions/runs", headers=headers, params={"workflow_id": workflow_id})
    response.raise_for_status()

    try:
        # Get the run ID of the latest run
        latest_run_id = response.json()["workflow_runs"][0]["id"]

        # Get the log URL for the latest run
        log_url = f"https://api.github.com/repos/{repository}/actions/runs/{latest_run_id}/logs"
        log_response = requests.get(log_url, headers=headers)
        log_response.raise_for_status()

        # Fetch the log content
        log_content = log_response.json()["content"]

        # Decode the log content
        decoded_content = base64.b64decode(log_content).decode("utf-8")

        # Print or process the log content as needed
        print(f"Workflow: {workflow_name}")
        # print(f"Workflow: {workflow_name}")
        print(f"Run ID: {latest_run_id}")
        print(f"Log Content:\n{decoded_content}")
        # print(decoded_content)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"The log for the latest workflow run of '{workflow_name}' was not found. Please check the workflow run ID.")
            print (log_url)
        else:
            print(f"HTTP error: {e}")
    except Exception as e:
        print(f"An error occurred for '{workflow_name}': {e}")
