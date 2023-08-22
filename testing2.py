import github
import requests
import os
import pandas
from github import Github, Auth
import sys

class dashboard():
    def __init__(self): 
        self.github_token = os.environ["GIT_TOKEN"]
        
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)

        self.repo = self.auth.get_repo("Konjarla-Vindya/son-azureml-oss-models")
        self.repo_full_name = self.repo.full_name
        self.dict = {"workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [], "updated_at": [], "status": [], "conclusion": [], "badge": []}
        
        self.workflow_path = ".github/workflows/"

    def workflow_last_run(self):
        workflows = self.repo.get_workflows()
        headers = {"Authorization": f"Bearer {self.github_token}",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "Accept": "application/vnd.github+json"}
        
        for workflow in workflows:
            workflow_temp = workflow.name.replace(".github/workflows/", "")
            print("workflow is this :  ", workflow)
            workflow_temp = workflow_temp + ".yml"
            # if workflow_name != "":
            #     continue

            workflow_name = workflow_temp.replace("/", "-")
            print("Final workflow_name : ", workflow_name)
            
            try:
                response = requests.get(f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}/runs", headers=headers)
                response.raise_for_status()  # Raise an error if the response status code is not successful
                
                runs = response.json()
                #print(runs)
                print("Type is : ",type(runs["workflow_runs"]))
                print("Length is this : ", len(runs["workflow_runs"]))
                if len(runs["workflow_runs"]) != 0:
                    #continue
                    lastrun = runs["workflow_runs"][0]
                    self.workflow_name_ext = lastrun["name"].replace(self.workflow_path, "")
                    badgeurl = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}/badge.svg"
                    id = lastrun['id']
                    job_id_url = f"https://github.com/{self.repo_full_name}/actions/runs/{id}/jobs"
                    job_details = requests.get(job_id_url, headers=headers)
                    job_detail = job_details.json()["jobs"][0]
                    job_id = job_detail["id"]
                    final_job_url = f"https://github.com/{self.repo_full_name}/actions/runs/{id}/job/{job_id}"

                    self.dict["workflow_id"].append(lastrun["workflow_id"])
                    self.dict["workflow_name"].append(self.workflow_name_ext.replace(".yml", ""))
                    self.dict["last_runid"].append(lastrun["id"])
                    self.dict["created_at"].append(lastrun["created_at"])
                    self.dict["updated_at"].append(lastrun["updated_at"])
                    self.dict["status"].append(lastrun["status"])
                    self.dict["conclusion"].append(lastrun["conclusion"])
                    #self.dict["badge"].append(f"[![{workflow_name}]({badgeurl})]({badgeurl.replace('/badge.svg', '')})")
                    self.dict["badge"].append(f"[![{workflow_name}]({final_job_url})]({badgeurl.replace('/badge.svg', '')})")

            except requests.exceptions.RequestException as e:
                _, _, exc_tb = sys.exc_info()
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")
                print(f"The exception occured at this line no : {exc_tb.tb_lineno} ")
            break


        return self.dict



    def results(self,last_runs_dict):
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
        summary.append(f"100.0%|{success_rate}|{failure_rate}|{cancel_rate}|")

        models = {"Model": last_runs_dict["workflow_name"],"Status": last_runs_dict["badge"]}
        models_md = pandas.DataFrame.from_dict(models).to_markdown()

        summary_text = ""
        for row in summary:
            summary_text += row + "\n"

        with open("testing2.md", "w", encoding="utf-8") as f:
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
