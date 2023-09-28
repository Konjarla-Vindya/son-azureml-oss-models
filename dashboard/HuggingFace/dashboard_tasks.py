import os,sys
import requests
import re
import pandas
from datetime import datetime
from github import Github, Auth
# from bs4 import BeautifulSoup

 

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
                     if workflow["name"].lower().startswith("mlflow-mp-") or workflow["name"].lower().startswith("mlflow-di-"):
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
         print(workflow_name)
         return workflow_name




    def workflow_last_run(self):   
        workflows_to_include = self.get_all_workflow_names()
        normalized_workflows = [workflow_name.replace("/","-") for workflow_name in workflows_to_include]
        workflow_actual_name = [workflow_actual_name for workflow_actual_name in workflows_to_include]
        workflow_actual_names = [name.replace("MLFlow-MP-", "").replace("MLFlow-DI-","") for name in workflow_actual_name]
        
        # Initialize counters for different statuses and categories
        success_count = 0
        failure_count = 0
        cancelled_count = 0
        running_count = 0
        not_tested_count = 0
    
        dynamic_installation_statuses = {
            "success": 0,
            "failure": 0,
            "cancelled": 0,
            "running": 0,
            "not_tested": 0
        }
    
        packaging_statuses = {
            "success": 0,
            "failure": 0,
            "cancelled": 0,
            "running": 0,
            "not_tested": 0
        }
    
        total_dynamic_installation_count = 0
        total_packaging_count = 0
        
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
    
                
    
                # Count statuses and categorize by category
                category = None
                if workflow_name.lower().startswith("mlflow-di-"):
                    category = "Online Endpoint Deployment - Dynamic Installation"
                    total_dynamic_installation_count += 1
                elif workflow_name.lower().startswith("mlflow-mp-"):
                    category = "Online Endpoint Deployment - Packaging"
                    total_packaging_count += 1
    
                if last_run["conclusion"] == "success":
                    if category == "Online Endpoint Deployment - Dynamic Installation":
                        dynamic_installation_statuses["success"] += 1
                    elif category == "Online Endpoint Deployment - Packaging":
                        packaging_statuses["success"] += 1
                    success_count += 1
                elif last_run["conclusion"] == "failure":
                    if category == "Online Endpoint Deployment - Dynamic Installation":
                        dynamic_installation_statuses["failure"] += 1
                    elif category == "Online Endpoint Deployment - Packaging":
                        packaging_statuses["failure"] += 1
                    failure_count += 1
                elif last_run["conclusion"] == "cancelled":
                    if category == "Online Endpoint Deployment - Dynamic Installation":
                        dynamic_installation_statuses["cancelled"] += 1
                    elif category == "Online Endpoint Deployment - Packaging":
                        packaging_statuses["cancelled"] += 1
                    cancelled_count += 1
                elif last_run["status"] == "in_progress":
                    running_count += 1
                else:
                    not_tested_count += 1
            except requests.exceptions.RequestException as e:
               print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")         
     # Calculate percentages
        success_percentage = (success_count / total_dynamic_installation_count) * 100 if total_dynamic_installation_count > 0 else 0
        failure_percentage = (failure_count / total_dynamic_installation_count) * 100 if total_dynamic_installation_count > 0 else 0
        cancelled_percentage = (cancelled_count / total_dynamic_installation_count) * 100 if total_dynamic_installation_count > 0 else 0
        running_percentage = (running_count / total_dynamic_installation_count) * 100 if total_dynamic_installation_count > 0 else 0
        not_tested_percentage = (not_tested_count / total_dynamic_installation_count) * 100 if total_dynamic_installation_count > 0 else 0
        summary = []
        summary.append("Category | Total Model | Pass | Pass % | Failure | Failure % | Cancelled | Running/In Progress | Not Tested|") 
        summary.append("-------- | ----------- | ---- | ------- | ------- | ---------- | --------- | ------------------- | ----------|")
        # summary.append(f"Online Endpoint Deployment - Dynamic Installation | {total_dynamic_installation_count} | {dynamic_installation_statuses["success"]} | {success_percentage:.2f}% | {dynamic_installation_statuses["failure"]} | {failure_percentage:.2f}% | {dynamic_installation_statuses["cancelled"]} | {dynamic_installation_statuses["running"]} | {dynamic_installation_statuses["not_tested"]}|")      
        summary.append(f"Online Endpoint Deployment - Dynamic Installation | {total_dynamic_installation_count} | {dynamic_installation_statuses['success']} | {(dynamic_installation_statuses['success'] / total_dynamic_installation_count) * 100}% | {dynamic_installation_statuses['failure']} | {failure_percentage:.2f}% | {dynamic_installation_statuses['cancelled']} | {dynamic_installation_statuses['running']} | {dynamic_installation_statuses['not_tested']}|")
        summary.append(f"Online Endpoint Deployment - Packaging | {total_packaging_count} | {packaging_statuses['success']} | {(packaging_statuses['success'] / total_packaging_count) * 100}% | {packaging_statuses['failure']} | {failure_percentage:.2f}% | {packaging_statuses['cancelled']} | {packaging_statuses['running']} | {packaging_statuses['not_tested']} |")
        # summary.append(f"Batch Endpoint Deployment | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"Finetune | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"Evaluation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"FT Model Online Endpoint Deployment - Dynamic Installation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"FT Model Online Endpoint Deployment - Packaging | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"FT Model Batch EndPoint Deployment | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"FT Model Evaluation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"Import | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        # summary.append(f"Inference with Parameters | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |")
        
        # #summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        # #summary.append("-----|-------|-------|-------|-------|")
        # summary.append(f"Online Endpoint Deployment - Dynamic Installation|{results_dict['total_di']}|{results_dict['success_di']}|{results_dict['failure_di']}|{results_dict['cancelled_di']}|{results_dict['running_di']}|")
        # summary.append(f"Online Endpoint Deployment - Model Packaging|{results_dict['total_mp']}|{results_dict['success_mp']}|{results_dict['failure_mp']}|{results_dict['cancelled_mp']}|{results_dict['running_mp']}|")
    
        # Create and print the matrix table with the "Category" column
        # matrix_table = f"""
        # |Category | Total Model | Pass | Pass % | Failure | Failure % | Cancelled | Running/In Progress | Not Tested|
        # |-------- | ----------- | ---- | ------- | ------- | ---------- | --------- | ------------------- | ----------|
        # |Online Endpoint Deployment - Dynamic Installation | {total_dynamic_installation_count} | {dynamic_installation_statuses["success"]} | {success_percentage:.2f}% | {dynamic_installation_statuses["failure"]} | {failure_percentage:.2f}% | {dynamic_installation_statuses["cancelled"]} | {dynamic_installation_statuses["running"]} | {dynamic_installation_statuses["not_tested"]}|
        # |Online Endpoint Deployment - Packaging | {total_packaging_count} | {packaging_statuses["success"]} | {success_percentage:.2f}% | {packaging_statuses["failure"]} | {failure_percentage:.2f}% | {packaging_statuses["cancelled"]} | {packaging_statuses["running"]} | {packaging_statuses["not_tested"]} |
        # |Batch Endpoint Deployment | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |Finetune | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |Evaluation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |FT Model Online Endpoint Deployment - Dynamic Installation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |FT Model Online Endpoint Deployment - Packaging | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |FT Model Batch EndPoint Deployment | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |FT Model Evaluation | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |Import | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # |Inference with Parameters | 0 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0 |
        # """
        summary_text = "\n".join(summary)
        print (summary_text)   
        return summary_text
           
    # def extract_error_messages(self, job_url):
    #     try:
    #         response = requests.get(job_url, headers={"Authorization": f"Bearer {self.github_token}", "Accept": "text/html"})
    #         response.raise_for_status()
    #         html_content = response.text
    #         print(html_content)
    #         print(job_url)
    #         # Parse the HTML content using BeautifulSoup
    #         soup = BeautifulSoup(html_content, "html.parser")
    
    #         # Find and extract both error and failure messages
    #         error_messages = []
    
    #         for paragraph in soup.find_all("p"):
    #             text = paragraph.get_text()
    #             # Check if the text contains common error or failure indicators
    #             if re.search(r'(raise error|raise|error|error message|failure message|\"message\":)', text, re.IGNORECASE):
    #                 # Truncate the message at the first occurrence of '\n'
    #                 first_newline_index = text.find('\n')
    #                 if first_newline_index != -1:
    #                     text = text[:first_newline_index]
    #                 error_messages.append(text.strip())  # Strip leading/trailing whitespace
    
    #         error_messages = "\n".join(error_messages)
    
    #         return error_messages
    
    #     except requests.exceptions.RequestException as e:
    #         print(f"An error occurred while fetching messages from '{job_url}': {e}")
    #         return "Error: Unable to fetch messages"
    

    def results(self, summary_text):
       
        # dashboard_tasks  =  workflow_last_run()
        
        with open("dashboard_tasks.md", "w", encoding="utf-8") as f:
            f.write(summary_text)
            

def main():

        my_class = Dashboard()
        summary_text = my_class.workflow_last_run()  
        my_class.results(summary_text)

if __name__ == "__main__":
    main()
