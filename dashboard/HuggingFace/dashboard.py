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
        self.repo = self.auth.get_repo("Konjarla-Vindya/son-azureml-oss-models")
        self.repo_full_name = self.repo.full_name
        self.data = {
            "workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [],
            "updated_at": [], "status": [], "conclusion": [], "jobs_url": []
        }
        self.models_data = []  # Initialize models_data as an empty list

    def get_all_workflow_names(self):
        # workflow_name = ["abc/def","abeja/gpt-neox-japanese-2.7b","abhishek/llama-2-7b-hf-small-shards","abnersampaio/sentiment","adamc-7/distilbert-imdb-micro"]
        API = "https://api.github.com/repos/Konjarla-Vindya/son-azureml-oss-models/actions/workflows"
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
                    if workflow["name"].lower().startswith("mlflow"):
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
        #create ../logs/get_github_workflows/ if it does not exist
        # if not os.path.exists("../logs/get_all_workflow_names"):
        #     os.makedirs("../logs/get_all_workflow_names")
        # # dump runs as json file in ../logs/get_github_workflows folder with filename as DDMMMYYYY-HHMMSS.json
        # with open(f"../logs/get_all_workflow_names/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        #     json.dump(workflow_name, f, indent=4)
        return workflow_name



    def workflow_last_run(self): 
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/", "-") for workflow_name in workflows_to_include]

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

 

                self.data["workflow_id"].append(last_run["workflow_id"])
                self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
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
                    # "Status": "<span style='background-color: #00FF00; padding: 2px 6px; border-radius: 3px;'>PASS</span>" if last_run["conclusion"] == "success" else "<span style='background-color: #FF0000; padding: 2px 6px; border-radius: 3px;'>FAIL</span>",
                    "Status": " ✅ PASS" if last_run["conclusion"] == "success" else "❌ FAIL",
                    "Link": f"[Run Link]({run_link})",
                    "LastRun_Timestamp": last_run["created_at"]
                }

                self.models_data.append(models_entry)

 

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

 

        models_df = pandas.DataFrame.from_dict(self.models_data)
        models_md = models_df.to_markdown()

 

        summary_text = "\n".join(summary)
        current_date = datetime.now().strftime('%Y%m%d')
    
        # Create a README file with the current datetime in the filename
        readme_filename = f"README_{current_date}.md"

 

        with open(readme_filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
            f.write(os.linesep)
            f.write(os.linesep)
            f.write(models_md)

        with open("README.md", "w", encoding="utf-8") as f:
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
