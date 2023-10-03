import os,sys
import requests
import pandas
from datetime import datetime
from github import Github, Auth

 

class Dashboard():
    def __init__(self): 
        self.github_token = os.environ['token']
        #self.github_token = "API_TOKEN"
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)
        self.repo = self.auth.get_repo("Azure/azure-ai-model-catalog")
        self.repo_full_name = self.repo.full_name
        self.data = {
            "workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [],
            "updated_at": [], "status": [], "conclusion": [], "jobs_url": []
        }
        self.models_data = []  # Initialize models_data as an empty list

    def get_all_workflow_names(self):
        # workflow_name = ["MLFlow-mosaicml/mpt-30b-instruct"]
        API = "https://api.github.com/repos/Azure/azure-ai-model-catalog/actions/workflows"
        print (f"Getting github workflows from {API}")
        total_pages = None
        current_page = 1
        per_page = 100
        workflow_name = []
        while total_pages is None or current_page <= total_pages:

            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            params = { "per_page": per_page, "page": current_page }
            response = requests.get(API, headers=headers, params=params)
            if response.status_code == 200:
                workflows = response.json()
                # append workflow_runs to runs list
                for workflow in workflows["workflows"]:
                    if (workflow["name"].startswith("MLFlow-MP") | workflow["name"].startswith("MLFlow-DI")):
                        workflow_name.append(workflow["name"])
                if not workflows["workflows"]:
                    break
                # workflow_name.extend(json_response['workflows["name"]'])
                if current_page == 1:
                # divide total_count by per_page and round up to get total_pages
                    total_pages = int(workflows['total_count'] / per_page) + 1
                current_page += 1
                # print a single dot to show progress
                print (f"\rWorkflows fetched: {len(workflow_name)}", end="", flush=True)
            else:
                print (f"Error: {response.status_code} {response.text}")
                exit(1)
        print (f"\n")
        # create ../logs/get_github_workflows/ if it does not exist
        # if not os.path.exists("../logs/get_all_workflow_names"):
        #     os.makedirs("../logs/get_all_workflow_names")
        # # dump runs as json file in ../logs/get_github_workflows folder with filename as DDMMMYYYY-HHMMSS.json
        # with open(f"../logs/get_all_workflow_names/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        #     json.dump(workflow_name, f, indent=4)
        return workflow_name



    def workflow_last_run(self): 
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/","-") for workflow_name in workflows_to_include]
        # normalized_workflows = [hf_name for hf_name in workflows_to_include]
        # hf_name = [hf_name for hf_name in workflows_to_include]
        #print(workflow_name)
        # print(hf_name)
        for workflow_name in normalized_workflows:
            try:
                workflow_runs_url = f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/runs"
                response = requests.get(workflow_runs_url, headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                response.raise_for_status()
                runs_data = response.json()

 

                if "workflow_runs" not in runs_data:
                    print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                    continue

 

                workflow_runs = runs_data["workflow_runs"]
                if not workflow_runs:
                    print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                    continue

 

                last_run = workflow_runs[0]
                jobs_response = requests.get(last_run["jobs_url"], headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                jobs_data = jobs_response.json()

 

               # badge_url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/badge.svg"
                html_url = jobs_data["jobs"][0]["html_url"] if jobs_data.get("jobs") else ""

 
                #self.data["workflow_name_mp"] = data["workflow_name"].startswith("MLFlow-MP") == True 
                #self.data["workflow_name_di"] = data["workflow_name"].startswith("MLFlow-DI") == True 
                self.data["workflow_id"].append(last_run["workflow_id"])
                self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                #self.data["workflow_name_di"].append(workflow_name.replace(".yml", ""))
                self.data["last_runid"].append(last_run["id"])
                self.data["created_at"].append(last_run["created_at"])
                self.data["updated_at"].append(last_run["updated_at"])
                self.data["status"].append(last_run["status"])
                self.data["conclusion"].append(last_run["conclusion"])
                self.data["jobs_url"].append(html_url)

 

                #if html_url:
                    #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({html_url})")
                #else:
                    #url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml"
                    #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({url})")
                run_link = f"https://github.com/{self.repo_full_name}/actions/runs/{last_run['id']}"
                models_entry = {
                    "Model": workflow_name.replace(".yml", ""),
                    # "HFLink": f"[Link](https://huggingface.co/{workflow_name.replace(".yml", "").replace("MLFlow-","")})",
                    # "Status": "<span style='background-color: #00FF00; padding: 2px 6px; border-radius: 3px;'>PASS</span>" if last_run["conclusion"] == "success" else "<span style='background-color: #FF0000; padding: 2px 6px; border-radius: 3px;'>FAIL</span>",
                    # "Status": " ✅ PASS" if last_run["conclusion"] == "success" elif last_run["conclusion"] == "failure" "❌ FAIL",
                    "Status": f"{'✅ PASS' if last_run['conclusion'] == 'success' else '❌ FAIL' if last_run['conclusion'] == 'failure' else '🚫 CANCELLED' if last_run['conclusion'] == 'cancelled' else '⏳ RUNNING'}",
                    "LastRunLink": f"[Link]({run_link})",
                    "LastRunTimestamp": last_run["created_at"],
                    "Model Package/Dynmaic Installation": f"""{'Model Package' if workflow_name.startswith("MLFlow-MP") == True else 'Dynmaic Installation' if workflow_name.startswith("MLFlow-DI") == True else 'None' }"""
                }

                self.models_data.append(models_entry)

 

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

 
        # self.models_data.sort(key=lambda x: x["Status"])
        self.models_data.sort(key=lambda x: (x["Status"] != "❌ FAIL", x["Status"]))
        return self.data

    def results(self, last_runs_dict):
        results_dict = {"total_mp": 0, "success_mp": 0, "failure_mp": 0, "cancelled_mp": 0,"running_mp":0, "not_tested_mp": 0, "total_duration_mp": 0,
                        "total_di": 0, "success_di": 0, "failure_di": 0, "cancelled_di": 0,"running_di":0, "not_tested_di": 0, "total_duration_di": 0}
        summary = []

 

        df = pandas.DataFrame.from_dict(last_runs_dict)
        # df = df.sort_values(by=['status'], ascending=['failure' in df['status'].values])
      
        results_dict["total_mp"] =  df.loc[df["workflow_name"].str.startswith("MLFlow-MP") == True]["workflow_id"].count()
        results_dict["success_mp"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()
        results_dict["failure_mp"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()
        results_dict["cancelled_mp"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()
        results_dict["running_mp"] = df.loc[(df['status'] == 'in_progress') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()  # Add running count
        results_dict["not_tested_mp"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("MLFlow-MP") == True)]['workflow_id'].count()

         
        results_dict["total_di"] = df.loc[df["workflow_name"].str.startswith("MLFlow-DI") == True]["workflow_id"].count()
        results_dict["success_di"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success') & (df["workflow_name"].str.startswith("MLFlow-DI") == True)]['workflow_id'].count()
        results_dict["failure_di"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure') & (df["workflow_name"].str.startswith("MLFlow-DI") == True)]['workflow_id'].count()
        results_dict["cancelled_di"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled') & (df["workflow_name"].str.startswith("MLFlow-DI") == True)]['workflow_id'].count()
        results_dict["running-di"] = df.loc[(df['status'] == 'in_progress')& (df["workflow_name"].str.startswith("MLFlow-DI") == True)]['workflow_id'].count()  # Add running count
        results_dict["not_tested_di"] = df.loc[(df['status'] != 'completed') & (df["workflow_name"].str.startswith("MLFlow-DI") == True)]['workflow_id'].count()



        success_rate_di = results_dict["success_di"]/results_dict["total_di"]*100.00
        failure_rate_di = results_dict["failure_di"]/results_dict["total_di"]*100.00
        cancel_rate_di = results_dict["cancelled_di"]/results_dict["total_di"]*100.00
        running_rate_di = results_dict["running_di"] / results_dict["total_di"] * 100.00  # Calculate running rate

        success_rate_mp = results_dict["success_mp"]/results_dict["total_mp"]*100.00
        failure_rate_mp = results_dict["failure_mp"]/results_dict["total_mp"]*100.00
        cancel_rate_mp = results_dict["cancelled_mp"]/results_dict["total_mp"]*100.00
        running_rate_mp = results_dict["running_mp"] / results_dict["total_mp"] * 100.00  # Calculate running rate

 
        
        summary.append("|Category|🚀Total|✅Pass|Pass%|❌Failure|Failure%|🚫Cancelled|⏳Running|❗️NotTested") 
        summary.append("| ----------- | ----------------- | -------- | -------- | --------  | -------- | --------- | ---------- | -----------|")
        #summary.append("| Online Endpoint Deployment - Dynamic Installation| ")      
        #summary.append("| Online Endpoint Deployment - Packaging| )
        #summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        #summary.append("-----|-------|-------|-------|-------|")
        summary.append(f"Online Endpoint Deployment - Dynamic Installation|{results_dict['total_di']}|{results_dict['success_di']}|{(results_dict['success_di']/results_dict['total_di'])*100}%|{results_dict['failure_di']}|{(results_dict['failure_di']/results_dict['total_di'])*100}%|{results_dict['cancelled_di']}|{results_dict['running_di']}|{results_dict['not_tested_di']}|")
        summary.append(f"Online Endpoint Deployment - Model Packaging|{results_dict['total_mp']}|{results_dict['success_mp']}|{(results_dict['success_mp']/results_dict['total_mp'])*100}%|{results_dict['failure_mp']}|{(results_dict['failure_mp']/results_dict['total_mp'])*100}%|{results_dict['cancelled_mp']}|{results_dict['running_mp']}|{results_dict['not_tested_mp']}|")

        models_df = pandas.DataFrame.from_dict(self.models_data)
        models_md = models_df.to_markdown()

 

        summary_text = "\n".join(summary)
       
        with open("dashboard_tasks.md", "w", encoding="utf-8") as f:
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
