import github
import requests
import os
import pandas
from github import Github, Auth

class dashboard():
    def __init__(self): 
        self.github_token = os.environ["GIT_TOKEN"]
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)
        self.repo = self.auth.get_repo("Konjarla-Vindya/son-azureml-oss-models")
        self.repo_full_name = self.repo.full_name
        self.dict = {"workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [], "updated_at": [], "status": [], "conclusion": [], "badge": []}
        
    def workflow_last_run(self):
        workflows = self.repo.get_workflows()
        headers = {"Authorization": f"Bearer {self.github_token}",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "Accept": "application/vnd.github+json"}



        for workflow in workflows:
            workflow_name = workflow.name.replace(".github/workflows/", "")
            workflow_name = workflow_name.replace("/", "-")
    
            if workflow_name in ["ahotrod-electra_large_discriminator_squad2_512.yml","testing.yml"]:
                continue

               
    
            try:
                response = requests.get("https://api.github.com/repos/{}/actions/workflows/{}/runs".format(self.repo_full_name, workflow_name), headers=headers)
                response.raise_for_status()  # Raise an error if the response status code is not successful
                runs = response.json()
                
                if not runs["workflow_runs"]: 
                    print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                    continue
                
                lastrun = runs["workflow_runs"][0]
                # 
                
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")
                continue  # Move to the next iteration of the loop

        

                
                # runs = response.json()
                # lastrun = runs["workflow_runs"][0]
                badgeurl = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow.name}/badge.svg"

                self.dict["workflow_id"].append(lastrun["workflow_id"])
                self.dict["workflow_name"].append(workflow.name.replace(".yml", ""))
                self.dict["last_runid"].append(lastrun["id"])
                self.dict["created_at"].append(lastrun["created_at"])
                self.dict["updated_at"].append(lastrun["updated_at"])
                self.dict["status"].append(lastrun["status"])
                self.dict["conclusion"].append(lastrun["conclusion"])
                self.dict["badge"].append(f"[![{workflow.name}]({badgeurl})]({badgeurl.replace('/badge.svg', '')})")

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow.name}' (ID: {workflow.id}): {e}")

        return self.dict

    def results(self, last_runs_dict):
        results_dict = {"total": 0, "success": 0, "failure": 0, "cancelled": 0, "not_tested": 0, "total_duration": 0}
        summary = []

        df = pandas.DataFrame.from_dict(last_runs_dict)
        results_dict["total"] = df["workflow_id"].count()
        results_dict["success"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success')]['workflow_id'].count()
        results_dict["failure"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure')]['workflow_id'].count()
        results_dict["cancelled"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled')]['workflow_id'].count()
    
        success_rate = results_dict["success"]/results_dict["total"]*100.00
        failure_rate = results_dict["failure"]/results_dict["total"]*100.00
        cancel_rate = results_dict["cancelled"]/results_dict["total"]*100.00

        summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|")
        summary.append("-----|-------|-------|-------|")
        summary.append(f"{results_dict['total']}|{results_dict['success']}|{results_dict['failure']}|{results_dict['cancelled']}|")
        summary.append(f"100.0%|{success_rate:.2f}%|{failure_rate:.2f}%|{cancel_rate:.2f}%|")

        models = {"Model": last_runs_dict["workflow_name"], "Status": last_runs_dict["badge"]}
        models_md = pandas.DataFrame.from_dict(models).to_markdown()

        summary_text = "\n".join(summary)

        with open("testing.md", "w", encoding="utf-8") as f:
            f.write(summary_text)
            f.write(os.linesep)
            f.write(os.linesep)
            f.write(models_md)

def main():
    my_class = dashboard()
    last_runs_dict = my_class.workflow_last_run()
    my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
