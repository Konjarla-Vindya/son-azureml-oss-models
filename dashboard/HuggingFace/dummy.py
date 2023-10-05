import requests
import os

# Set your GitHub username, repository name, workflow name, and personal access token
username = "Konjarla-Vindya"
repository = "son-azureml-oss-models"
workflow_name = "dashboard.yml"
access_token = os.environ['token']

# Define the API endpoint
api_url = f"https://api.github.com/repos/{username}/{repository}/actions/workflows/{workflow_name}/runs"

# Send a GET request to fetch the workflow runs
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/vnd.github.v3+json"
}
response = requests.get(api_url, headers=headers)

if response.status_code == 200:
    data = response.json()
    if data.get("workflow_runs"):
        latest_run = data["workflow_runs"][0]  # Get the latest run
        for job in latest_run["jobs"]:
            if job["status"] == "failure":
                # Download job log
                job_id = job["id"]
                job_name = job["name"]
                log_url = f"{api_url}/{latest_run['id']}/jobs/{job_id}/logs/{job_name}.log"
                log_response = requests.get(log_url, headers=headers)
                if log_response.status_code == 200:
                    log_content = log_response.text
                    print(log_content)
                    # Save the log to a file
                    # with open(f"{job_name}_log.txt", "w") as log_file:
                    #     log_file.write(log_content)
                else:
                    print(f"Failed to download job log for {job_name}")
else:
    print(f"Failed to fetch workflow runs. Status code: {response.status_code}")
