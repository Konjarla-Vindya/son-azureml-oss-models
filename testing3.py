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
        
    def workflow_last_run(self):
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json"
        }
        
        workflows_to_include = [
            "CAMeL-Lab-bert-base-arabic-camelbert-ca-sentiment.yml",
            "Helsinki-NLP-opus-mt-trk-en.yml",
            "Helsinki-NLP-opus-mt-en-bem.yml",
            "Helsinki-NLP-opus-mt-it-bg.yml",
            "SauravMaheshkar-clr-finetuned-albert-base.yml",
            "DoyyingFace-bert-asian-hate-tweets-asian-unclean-freeze-4.yml",
            "Helsinki-NLP-opus-mt-en-bi.yml",
            "Helsinki-NLP-opus-mt-ja-pt.yml",
            "Helsinki-NLP-opus-mt-en-bzs.yml",
            "Helsinki-NLP-opus-mt-en-ht.yml",
            "Geotrend-bert-base-uk-cased.yml",
            "Helsinki-NLP-opus-mt-sv-uk.yml",
            "Helsinki-NLP-opus-mt-es-zai.yml",
            "EasthShin-Android_Ios_Classification.yml",
            "Helsinki-NLP-opus-mt-en-lua.yml",

         
            "ahotrod-electra_large_discriminator_squad2_512.yml",
            "ARTeLab-it5-summarization-fanpage.yml"
        ]

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
                runurl = "https://github.com/{}/actions/runs/{}/job/{}".format(self.repo_full_name,lastrun["id"],job["jobs"][0]["id"])

                self.data["workflow_id"].append(lastrun["workflow_id"])
                self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                self.data["last_runid"].append(lastrun["id"])
                self.data["created_at"].append(lastrun["created_at"])
                self.data["updated_at"].append(lastrun["updated_at"])
                self.data["status"].append(lastrun["status"])
                self.data["conclusion"].append(lastrun["conclusion"])
                #self.data["badge"].append(f"[![{workflow_name}]({badgeurl})]({badgeurl.replace('/badge.svg', '')})")
                self.data["badge"].append("[![{}]({})]({})".format(self.workflow_name_ext,self.badgeurl,self.runurl ))
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
