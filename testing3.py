import os
import requests
import pandas
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
            "updated_at": [], "status": [], "conclusion": [], "badge": []
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
                response = requests.get(f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}/runs", headers=headers)
                response.raise_for_status()
                
                runs = response.json()
                if not runs["workflow_runs"]: 
                    print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                    continue
                
                lastrun = runs["workflow_runs"][0]
                badgeurl = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}/badge.svg"

                self.data["workflow_id"].append(lastrun["workflow_id"])
                self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                self.data["last_runid"].append(lastrun["id"])
                self.data["created_at"].append(lastrun["created_at"])
                self.data["updated_at"].append(lastrun["updated_at"])
                self.data["status"].append(lastrun["status"])
                self.data["conclusion"].append(lastrun["conclusion"])
                self.data["badge"].append(f"[![{workflow_name}]({badgeurl})]({badgeurl.replace('/badge.svg', '')})")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

        return self.data

    # ... rest of the class methods ...

def main():
    my_class = Dashboard()
    last_runs_dict = my_class.workflow_last_run()
    my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
