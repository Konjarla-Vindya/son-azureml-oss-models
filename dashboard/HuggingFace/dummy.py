import requests
import os

# Set your GitHub username, repository name, workflow name, and personal access token
username = "Konjarla-Vindya"
repository = "son-azureml-oss-models"
workflow_name = "distilbert-base-cased-distilled-squad.yml"
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
        api = "https://api.github.com/repos/Konjarla-Vindya/son-azureml-oss-models/actions/runs/6205287620/logs"
       


        # Send a GET request to fetch job details
        job_response = requests.get(job_api_url, headers=headers)
        api_response = requests.get(api, headers=headers)
        print("Job Response Status Code:", job_response.status_code)  # Debugging statement
        print("API Response Status Code:", api_response.status_code)  # Debugging statement
        if api_response.status_code == 200:
            api_data = api_response.json()
            print("Log Data:", api_data)  # Debugging statement
        if job_response.status_code == 200:
            job_data = job_response.json()
            print("Job Data:", job_data)  # Debugging statement
            for job in job_data["jobs"]:
                if job["status"] == "completed" and job["conclusion"] == "failure":
                    job_id = job["id"]
                    job_name = job["name"]
                    # Download job log
                    log_url = f"{job_api_url}/{job_id}/logs"
                    log_response = requests.get(log_url, headers=headers)
                    print("Log Response Status Code:", log_response.status_code)  # Debugging statement
                    if log_response.status_code == 200:
                        log_content = log_response.text
                        print("logs :", log_content)
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