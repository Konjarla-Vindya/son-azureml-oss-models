import os, sys
import requests
import pandas as pd
from github import Github, Auth

class Dashboard():
    def __init__(self): 
        self.github_token = os.environ["GIT_TOKEN"]
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)
        self.repo = self.auth.get_repo("Konjarla-Vindya/son-azureml-oss-models")
        self.repo_full_name = self.repo.full_name
        self.data = {
            "workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [],
            "updated_at": [], "status": [], "conclusion": [], "badge": [], "jobs_url": []
        }
        
    def get_all_workflow_names(self):
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get(f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows", headers=headers)
        response.raise_for_status()
        
        workflows = response.json()
        workflow_names = [workflow["name"] for workflow in workflows["workflows"]]
        print(workflow_names)
        return workflow_names
        
    def workflow_last_run(self):
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json"
        }
        
        workflows_to_include = self.get_all_workflow_names()

        for workflow_name in workflows_to_include:
            try:
                workflow_runs_url = f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}/runs"
                page = 1
                per_page = 100

                while True:
                    params = {
                        "page" : page,
                        "per_page" : per_page
                
                    response = requests.get(workflow_runs_url, headers=headers, params=params)
                    response.raise_for_status()
                
                    runs = response.json()["workflow_runs"]
                    if not runs: 
                        print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                        continue
                
                    lastrun = runs[0]
                    jobs_url = f"https://api.github.com/repos/{self.repo_full_name}/actions/runs/{lastrun['id']}/jobs"
                    jobresponse = requests.get(jobs_url)
                    job = jobresponse.json()["jobs"][0] if jobresponse.json()["jobs"] else {}

                    badgeurl = f"https://github.com/{self.repo_full_name}/workflows/{workflow_name}/badge.svg"
                    runurl = f"https://github.com/{self.repo_full_name}/actions/runs/{lastrun['id']}"
                    html_url = job.get("html_url", runurl)

                    self.data["workflow_id"].append(lastrun["workflow_id"])
                    self.data["workflow_name"].append(workflow_name)
                    self.data["last_runid"].append(lastrun["id"])
                    self.data["created_at"].append(lastrun["created_at"])
                    self.data["updated_at"].append(lastrun["updated_at"])
                    self.data["status"].append(lastrun["status"])
                    self.data["conclusion"].append(lastrun["conclusion"])
                    self.data["badge"].append(f"[![{workflow_name}]({badgeurl})]({html_url})")
                    self.data["jobs_url"].append(html_url)
                
                page+=1
              
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

        return self.data

    def results(self, last_runs_dict):
        results_dict = {"total": 0, "success": 0, "failure": 0, "cancelled": 0}
        summary = []
        df = pd.DataFrame.from_dict(last_runs_dict)  
        results_dict["total"] = df.shape[0]  # Get the total number of rows (workflow runs)
        if results_dict["total"] > 0: 
            results_dict["success"] = df.loc[df['conclusion'] == 'success'].shape[0]
            results_dict["failure"] = df.loc[df['conclusion'] == 'failure'].shape[0]
            results_dict["cancelled"] = df.loc[df['conclusion'] == 'cancelled'].shape[0]
            success_rate = results_dict["success"] / results_dict["total"] * 100.00
            failure_rate = results_dict["failure"] / results_dict["total"] * 100.00
            cancel_rate = results_dict["cancelled"] / results_dict["total"] * 100.00
        else:
            success_rate = 0.0 
            failure_rate = 0.0
            cancel_rate = 0.0

        summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|")
        summary.append("-----|-------|-------|-------|")
        summary.append(f"{results_dict['total']}|{results_dict['success']}|{results_dict['failure']}|{results_dict['cancelled']}|")
        summary.append(f"100.0%|{success_rate:.2f}%|{failure_rate:.2f}%|{cancel_rate:.2f}%|")
    
        models = {"Model": last_runs_dict["workflow_name"], "Badge": last_runs_dict["badge"]}
        models_df = pd.DataFrame.from_dict(models)

        with open("testing.md", "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
            f.write("\n\n")
            f.write(models_df.to_markdown(index=False))
def main():
    my_class = Dashboard()
    last_runs_dict = my_class.workflow_last_run()
    my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
