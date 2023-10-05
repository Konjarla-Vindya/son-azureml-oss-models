import requests
import os

# Set your GitHub username, repository name, workflow name, and personal access token
username = "Konjarla-Vindya"
repository = "son-azureml-oss-models"
workflow_name = "dashboard.yml"
access_token = os.environ['token']


# Define the API endpoint to fetch the latest workflow run
api_url = f"https://api.github.com/repos/{username}/{repository}/actions/workflows/{workflow_name}/runs"

# Send a GET request to fetch the latest workflow run
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/vnd.github.v3+json"
}
response = requests.get(api_url, headers=headers)

if response.status_code == 200:
    data = response.json()
    if data.get("workflow_runs"):
        latest_run = data["workflow_runs"][0]  # Get the latest run
        run_id = latest_run["id"]
        
        # Define the API endpoint to fetch job details for the latest run
        job_api_url = f"https://api.github.com/repos/{username}/{repository}/actions/runs/{run_id}/jobs"

        # Send a GET request to fetch job details
        job_response = requests.get(job_api_url, headers=headers)
        if job_response.status_code == 200:
            job_data = job_response.json()
            for job in job_data["jobs"]:
                if job["status"] == "completed" and job["conclusion"] == "failure":
                    job_id = job["id"]
                    job_name = job["name"]
                    # Download job log
                    log_url = f"{job_api_url}/{job_id}/logs"
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
            print(f"Failed to fetch job details. Status code: {job_response.status_code}")
    else:
        print("No workflow runs found.")
else:
    print(f"Failed to fetch workflow runs. Status code: {response.status_code}")
