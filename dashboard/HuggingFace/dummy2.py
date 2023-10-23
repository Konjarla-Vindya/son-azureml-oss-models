import os,sys
import requests
import pandas
import csv
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
         file_path = "tests/config/modellist.csv"  # Update this with the actual path
         try:
             url = f"https://raw.githubusercontent.com/{self.repo_full_name}/master/{file_path}"
             response = requests.get(url)
             response.raise_for_status()
             
             # Parse the CSV content and return it as a list
             csv_data = response.text.splitlines()
             csv_reader = csv.reader(csv_data)
             
             # Assuming the first column contains the data you want to retrieve
             mlflow_prefixed_data = ["MLFlow-" + row[0] for row in csv_reader]
             print(mlflow_prefixed_data)
             return mlflow_prefixed_data
             
             
         except Exception as e:
             print(f"Error fetching or parsing content from GitHub: {e}")
             return []
        # API = "https://api.github.com/repos/Azure/azure-ai-model-catalog/actions/workflows"
        # print (f"Getting github workflows from {API}")
        # total_pages = None
        # current_page = 1
        # per_page = 100
        # workflow_name = []
        # while total_pages is None or current_page <= total_pages:

        #     headers = {
        #         "Authorization": f"Bearer {self.github_token}",
        #         "Accept": "application/vnd.github.v3+json"
        #     }
        #     params = { "per_page": per_page, "page": current_page }
        #     response = requests.get(API, headers=headers, params=params)
        #     if response.status_code == 200:
        #         workflows = response.json()
        #         # append workflow_runs to runs list
        #         for workflow in workflows["workflows"]:
        #             if workflow["name"].lower().startswith("mlflow"):
        #                 workflow_name.append(workflow["name"])
        #         if not workflows["workflows"]:
        #             break
        #         # workflow_name.extend(json_response['workflows["name"]'])
        #         if current_page == 1:
        #         # divide total_count by per_page and round up to get total_pages
        #             total_pages = int(workflows['total_count'] / per_page) + 1
        #         current_page += 1
        #         # print a single dot to show progress
        #         print (f"\rWorkflows fetched: {len(workflow_name)}", end="", flush=True)
        #     else:
        #         print (f"Error: {response.status_code} {response.text}")
        #         exit(1)
        # print (f"\n")
        # # create ../logs/get_github_workflows/ if it does not exist
        # # if not os.path.exists("../logs/get_all_workflow_names"):
        # #     os.makedirs("../logs/get_all_workflow_names")
        # # # dump runs as json file in ../logs/get_github_workflows folder with filename as DDMMMYYYY-HHMMSS.json
        # # with open(f"../logs/get_all_workflow_names/{datetime.now().strftime('%d%b%Y-%H%M%S')}.json", "w") as f:
        # #     json.dump(workflow_name, f, indent=4)
        # return workflow_name


    def workflow_last_run(self): 
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/","-") for workflow_name in workflows_to_include]
        workflow_actual_name = [workflow_actual_name for workflow_actual_name in workflows_to_include]
        workflow_actual_names = [name.replace("MLFlow-", "") for name in workflow_actual_name]
        for workflow_actual_name, workflow_name in zip(workflow_actual_names, normalized_workflows):
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
                job_url = jobs_data["jobs"][0]["html_url"]
                

 

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
                # job_link  = f"https://github.com/{self.repo_full_name}/actions/runs/{last_run['id']}/jobs/"
                HF_Link = f"https://huggingface.co/{workflow_actual_name}"
                models_entry = {
                    "Model": workflow_actual_name,
                    # "HF_Link": f"[Link]({HF_Link})",
                    "Status": f"{'✅ PASS' if last_run['conclusion'] == 'success' else '❌ FAIL' if last_run['conclusion'] == 'failure' else '🚫 CANCELLED' if last_run['conclusion'] == 'cancelled' else '⏳ RUNNING'}",
                    # "LastRunLink": f"[Link]({run_link})",
                    # "LastRunTimestamp": last_run["created_at"],
                    
                }

                self.models_data.append(models_entry)

 

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

 
        # self.models_data.sort(key=lambda x: x["Status"])
        # self.models_data.sort(key=lambda x: (x["Status"] != "❌ FAIL", x["Status"]))
        return self.data
    
    def results(self, last_runs_dict):
        results_dict = {"total": 0, "success": 0, "failure": 0, "cancelled": 0,"running":0, "not_tested": 0, "total_duration": 0}
        summary = []
        failed_models = []

 

        df = pandas.DataFrame.from_dict(last_runs_dict)
        # df = df.sort_values(by=['status'], ascending=['failure' in df['status'].values])
        # results_dict["total"] = df["workflow_id"].count()
        # results_dict["success"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success')]['workflow_id'].count()
        # results_dict["failure"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure')]['workflow_id'].count()
        # results_dict["cancelled"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled')]['workflow_id'].count()
        # results_dict["running"] = df.loc[df['status'] == 'in_progress']['workflow_id'].count()  # Add running count


        # success_rate = results_dict["success"]/results_dict["total"]*100.00
        # failure_rate = results_dict["failure"]/results_dict["total"]*100.00
        # cancel_rate = results_dict["cancelled"]/results_dict["total"]*100.00
        # running_rate = results_dict["running"] / results_dict["total"] * 100.00  # Calculate running rate

 

        # summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        # summary.append("-----|-------|-------|-------|-------|")
        # summary.append(f"{results_dict['total']}|{results_dict['success']}|{results_dict['failure']}|{results_dict['cancelled']}|{results_dict['running']}|")
        # summary.append(f"100.0%|{success_rate:.2f}%|{failure_rate:.2f}%|{cancel_rate:.2f}%|{running_rate:.2f}%|")

 

        models_df = pandas.DataFrame.from_dict(self.models_data)
        failed_models_df = models_df[models_df['Status'] == '❌ FAIL']  # Filter only the failed models
        failed_models_list = failed_models_df['Model'].tolist()
        # models_md = failed_models_df.to_markdown()
        models_df = failed_models_df[['Model']]

        # Convert the filtered DataFrame to Markdown
        models_md = models_df.to_markdown(index=False)

 

        summary_text = "\n".join(summary)
        current_date = datetime.now().strftime('%Y%m%d')
    
        # Create a README file with the current datetime in the filename
        # readme_filename = f"README_{current_date}.md"
        # if not failed_models_df.empty:
        #     failed_models_list = failed_models_df['Model'].tolist()
            
        #     # Create a new file with the list of failed model names
        #     with open("failed_models.txt", "w", encoding="utf-8") as f:
        #         f.write("\n".join(failed_models_list))

 

        # with open(readme_filename, "w", encoding="utf-8") as f:
        #     f.write(summary_text)
        #     f.write(os.linesep)
        #     f.write(os.linesep)
        #     f.write(models_md)

        with open("failed_models.txt", "w", encoding="utf-8") as f:
            # f.write(summary_text)
            # f.write(os.linesep)
            # f.write(os.linesep)
            f.write("\n".join(failed_models_list))

 

def main():

        my_class = Dashboard()
        last_runs_dict = my_class.workflow_last_run()
        my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
