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

    def get_all_workflow_names(self,limit=50):
        #workflow_name = ["MLFlow-codellama/CodeLlama-13b-Instruct-hf","MLFlow-mosaicml/mpt-7b-storywriter","MLFlow-microsoft/MiniLM-L12-H384-uncased"]
        API = "https://api.github.com/repos/Azure/azure-ai-model-catalog/actions/workflows"
        print (f"Getting github workflows from {API}")
        # total_pages = None
        # current_page = 1
        # per_page = 100
        workflow_name = []
        # while total_pages is None or current_page <= total_pages:

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        params = { "per_page": limit}
        response = requests.get(API, headers=headers, params=params)
        if response.status_code == 200:
            workflows = response.json()
            # append workflow_runs to runs list
            for workflow in workflows["workflows"]:
                if workflow["name"].lower().startswith("mlflow"):
                    workflow_name.append(workflow["name"])
            # if not workflows["workflows"]:
            #     break
            # workflow_name.extend(json_response['workflows["name"]'])
            # if current_page == 1:
            # # divide total_count by per_page and round up to get total_pages
            #     total_pages = int(workflows['total_count'] / per_page) + 1
            # current_page += 1
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
        print(workflow_name)
        return workflow_name



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
                error_messages = self.extract_error_messages(job_url)

 

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
                    "HF_Link": f"[Link]({HF_Link})",
                    "Status": f"{'✅ PASS' if last_run['conclusion'] == 'success' else '❌ FAIL' if last_run['conclusion'] == 'failure' else '🚫 CANCELLED' if last_run['conclusion'] == 'cancelled' else '⏳ RUNNING'}",
                    "LastRunLink": f"[Link]({run_link})",
                    "LastRunTimestamp": last_run["created_at"],
                    "Error Message": error_messages
                }

                self.models_data.append(models_entry)

 

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

 
        # self.models_data.sort(key=lambda x: x["Status"])
        self.models_data.sort(key=lambda x: (x["Status"] != "❌ FAIL", x["Status"]))
        return self.data

    def extract_error_messages(self, job_url):
    try:
        response = requests.get(job_url, headers={"Authorization": f"Bearer {self.github_token}", "Accept": "text/html"})
        response.raise_for_status()
        html_content = response.text

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find and extract both error and failure messages
        error_messages = []

        for paragraph in soup.find_all("p"):
            text = paragraph.get_text()
            # Check if the text contains common error or failure indicators
            if re.search(r'(raise error|raise|error|error message|failure message)', text, re.IGNORECASE):
                # Truncate the message at the first occurrence of '\n'
                first_newline_index = text.find('\n')
                if first_newline_index != -1:
                    text = text[:first_newline_index]
                error_messages.append(text.strip())  # Strip leading/trailing whitespace

        combined_messages = "\n".join(error_messages)

        return combined_messages

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching messages from '{job_url}': {e}")
        return "Error: Unable to fetch messages"
    

    def results(self, last_runs_dict):
        results_dict = {"total": 0, "success": 0, "failure": 0, "cancelled": 0,"running":0, "not_tested": 0, "total_duration": 0}
        summary = []

 

        df = pandas.DataFrame.from_dict(last_runs_dict)
        # df = df.sort_values(by=['status'], ascending=['failure' in df['status'].values])
        results_dict["total"] = df["workflow_id"].count()
        results_dict["success"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success')]['workflow_id'].count()
        results_dict["failure"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure')]['workflow_id'].count()
        results_dict["cancelled"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled')]['workflow_id'].count()
        results_dict["running"] = df.loc[df['status'] == 'in_progress']['workflow_id'].count()  # Add running count


        success_rate = results_dict["success"]/results_dict["total"]*100.00
        failure_rate = results_dict["failure"]/results_dict["total"]*100.00
        cancel_rate = results_dict["cancelled"]/results_dict["total"]*100.00
        running_rate = results_dict["running"] / results_dict["total"] * 100.00  # Calculate running rate

 

        summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        summary.append("-----|-------|-------|-------|-------|")
        summary.append(f"{results_dict['total']}|{results_dict['success']}|{results_dict['failure']}|{results_dict['cancelled']}|{results_dict['running']}|")
        summary.append(f"100.0%|{success_rate:.2f}%|{failure_rate:.2f}%|{cancel_rate:.2f}%|{running_rate:.2f}%|")

 

        models_df = pandas.DataFrame.from_dict(self.models_data)
        models_md = models_df.to_markdown()

 

        summary_text = "\n".join(summary)
        current_date = datetime.now().strftime('%Y%m%d')
    
        # Create a README file with the current datetime in the filename
        readme_filename = f"README_{current_date}.md"

 

        # with open(readme_filename, "w", encoding="utf-8") as f:
        #     f.write(summary_text)
        #     f.write(os.linesep)
        #     f.write(os.linesep)
        #     f.write(models_md)

        with open("README_MI-DI.md", "w", encoding="utf-8") as f:
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
