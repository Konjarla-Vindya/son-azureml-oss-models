import os
import requests
import pandas
from github import Github, Auth

class Dashboard():
    def __init__(self): 
        self.github_token = "API_TOKEN"
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
        limit=50
        offset=0
        results_len = 1
        workflow_name = []
        while results_len != 0:
            

            # Set the parameters in the URL.
                params = {'per_page': limit, 'page': offset//limit+1}
            
                # Make the request combining the endpoint, headers and params above.
                #r = requests.get(endpoint, headers=headers, params=params)
                response = requests.get(f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows", headers=headers, params=params)
                response.raise_for_status()
                # Capture the results
                #print "Getting results for {}".format(r.url)
                #results = r.json()['Results']
                workflows = response.json()
                # We append all the results to the all_calls array.
                # for result in results:
                #     all_calls.append(result)
                for workflow in workflows["workflows"]:
                    workflow_name.append(workflow["name"])
                if not workflows["workflows"]:
                    break
            
                # Set the next limit.
                offset+=limit
        
            # If this is 0, we'll exit the while loop.
                results_len = len(workflows["workflows"]) 
        # response = requests.get(f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows?per_page=50", headers=headers)
        # response.raise_for_status()
        
        # workflows = response.json()
        # workflow_name = [workflow["name"] for workflow in workflows["workflows"]]
        print(workflow_name)
        return workflow_name
        
    def workflow_last_run(self):
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json"
        }
        
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/", "-") for workflow_name in workflows_to_include]


        for workflow_name in normalized_workflows:
            try:
                workflow_runs = f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/runs"
                response = requests.get(workflow_runs, headers=headers)
                response.raise_for_status()
                
                runs = response.json()
                if not runs["workflow_runs"]: 
                    print(f"No runs found for workflow '{workflow_names}'. Skipping...")
                    continue
                else:
                #if len(runs["workflow_runs"]) != 0:
                    lastrun = runs["workflow_runs"][0]
                    #URL_1 = f"https://api.github.com/repos/{self.repo_full_name}/actions/runs/{lastrun['id']}/jobs"
                    jobresponse = requests.get(lastrun["jobs_url"]) 
                    print("URL : ",lastrun["jobs_url"])
                    #print("URL : ",url)
                    job = jobresponse.json()
                    print(job)
                    
                    badgeurl = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/badge.svg"
                    #https://github.com/Konjarla-Vindya/son-azureml-oss-models/actions/workflows/TRIGGER_TESTS.yml/badge.svg
                    #runurl = "https://github.com/{}/actions/runs/{}/job/{}".format(self.repo_full_name,lastrun["id"],job["jobs"][0]["id"])
                    html_url=""
                    if len(job["jobs"])!=0:
                      html_url = job["jobs"][0]["html_url"]
            
                    
                    self.data["workflow_id"].append(lastrun["workflow_id"])
                    self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                    self.data["last_runid"].append(lastrun["id"])
                    self.data["created_at"].append(lastrun["created_at"])
                    self.data["updated_at"].append(lastrun["updated_at"])
                    self.data["status"].append(lastrun["status"])
                    self.data["conclusion"].append(lastrun["conclusion"])
                    self.data["jobs_url"].append(html_url)
                    #self.data["badge"].append(f"[![{workflow_name}]({badgeurl})]({badgeurl.replace('/badge.svg', '')})")
                    if len(html_url)!=0:
                        self.data["badge"].append("[![{}]({})]({})".format(workflow_name,badgeurl,html_url))
                        
                    else:
                        #f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/runs"
                        url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml"
                        self.data["badge"].append("[![{}]({})({})]".format(workflow_name,badgeurl,url))
                        
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

        return self.data

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
        
        my_class = Dashboard()
        last_runs_dict = my_class.workflow_last_run()
        my_class.results(last_runs_dict)
        
if __name__ == "__main__":
    main()
