import time, json, os 
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings
)

class ModelInferenceAndDeployemnt:
    def __init__(self, test_model_name, workspace_ml_client, registry_ml_client, registry) -> None:
        self.test_model_name = test_model_name
        self.workspace_ml_client = workspace_ml_client
        self.registry_ml_client = registry_ml_client
        self.registry = registry


    def get_latest_model_version(self, registry_ml_client, model_name):
        print ("In get_latest_model_version...")
        model_versions=registry_ml_client.models.list(name=model_name)
        #version_list = list(ml_client.models.list(model_name))
        model_version_count=0
        models = []
        for model in model_versions:
            model_version_count = model_version_count + 1
            models.append(model)
        sorted_models = sorted(models, key=lambda x: x.creation_context.created_at, reverse=True)
        latest_model = sorted_models[0]
        # print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}") 
        # print(latest_model)
        #latest_model = ""
        # model_version = model_versions[0].version
        # latest_model = model_versions.models.get(model_name, model_version)
        #     #return latest_model
        # return latest_model
        # model_list=list(registry_ml_client.models.list(name=model_name))
        # version = []
        # for model in model_list:
        #     version.append(model.version)
        # version = sorted(version, reverse=True)
        # latest_model = registry_ml_client.models.get(model_name, version[0])
        print (f"Latest model {latest_model.name} version {latest_model.version} created at {latest_model.creation_context.created_at}")
        print(f"Model Config : {latest_model.config}") 
        return latest_model
    
    def sample_inference(self, latest_model, registry, workspace_ml_client, online_endpoint_name):
        # get the task tag from the latest_model.tags
        tags = str(latest_model.tags)
        # replace single quotes with double quotes in tags
        tags = tags.replace("'", '"')
        # convert tags to dictionary
        tags_dict=json.loads(tags)
        task = tags_dict['task']
        print (f"task: {task}")
        scoring_file = f"../../config/sample_inputs/{registry}/{task}.json"
        # check of scoring_file exists
        try:
            with open(scoring_file) as f:
                scoring_input = json.load(f)
                print (f"scoring_input file:\n\n {scoring_input}\n\n")
        except Exception as e:
            print (f"::warning:: Could not find scoring_file: {scoring_file}. Finishing without sample scoring: \n{e}")

        # invoke the endpoint
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=online_endpoint_name,
                deployment_name="demo",
                request_file=scoring_file,
            )
            response_json = json.loads(response)
            output = json.dumps(response_json, indent=2)
            print(f"response: \n\n{output}")
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as fh:
                print(f'####Sample input', file=fh)
                print(f'```json', file=fh)
                print(f'{scoring_input}', file=fh)
                print(f'```', file=fh)
                print(f'####Sample output', file=fh)
                print(f'```json', file=fh)
                print(f'{output}', file=fh)
                print(f'```', file=fh)
        except Exception as e:
            print (f"::error:: Could not invoke endpoint: \n")
            print (f"{e}\n\n check logs:\n\n")

    def create_online_endpoint(self, workspace_ml_client, endpoint):
        print ("In create_online_endpoint...")
        try:
            workspace_ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        except Exception as e:
            print (f"::error:: Could not create endpoint: \n")
            print (f"{e}\n\n check logs:\n\n")
            prase_logs(str(e))
            exit (1)

        print(workspace_ml_client.online_endpoints.get(name=endpoint.name))


    def create_online_deployment(self, workspace_ml_client, endpoint, latest_model):
        print ("In create_online_deployment...")
        # demo_deployment = ManagedOnlineDeployment(
        #     name="demo",
        #     endpoint_name=endpoint.name,
        #     model=latest_model.id,
        #     instance_type="Standard_DS4_v2",
        #     instance_count=1,
        # )
        demo_deployment = ManagedOnlineDeployment(
            name="default",
            endpoint_name=endpoint.name,
            model="your_model_id",
            instance_type="instance_type",
            instance_count="1",
            request_settings=OnlineRequestSettings(
                max_concurrent_requests_per_instance=1,
                request_timeout_ms=50000,
                max_queue_wait_ms=500,
            ),
            liveness_probe=ProbeSettings(
                failure_threshold=10,
                timeout=10,
                period=10,
                initial_delay=480,
            ),
            readiness_probe=ProbeSettings(
                failure_threshold=10,
                success_threshold=1,
                timeout=10,
                period=10,
                initial_delay=10,
            ),
        )
        workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
        endpoint.traffic = {"demo": 100}
        workspace_ml_client.begin_create_or_update(endpoint).result()

        
    
    def model_infernce_deployment(self):
        latest_model = self.get_latest_model_version(self.registry_ml_client, self.test_model_name)
        #download_and_register_model()
        # get the task tag from the latest_model.tags
        # tags = str(latest_model.tags)
        # # replace single quotes with double quotes in tags
        # tags = tags.replace("'", '"')
        # # convert tags to dictionary
        # tags_dict=json.loads(tags)
        # print("the tags_dict is this one :",task)
        # task = tags_dict['task']
        # print("the task is:",task)
        # endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
        # timestamp = int(time.time())
        # #online_endpoint_name = task + str(timestamp)
        # online_endpoint_name = "Testing" + str(timestamp)
        # print (f"online_endpoint_name: {online_endpoint_name}")
        # endpoint = ManagedOnlineEndpoint(
        #     name=online_endpoint_name,
        #     auth_mode="key",
        # )
    
        # print("latest_model:",latest_model)
        # print("endpoint name:",endpoint)
        # self.create_online_endpoint(self.workspace_ml_client, endpoint)
        #self.create_online_deployment(self.workspace_ml_client, endpoint, latest_model)
        #self.sample_inference(latest_model, self.registry, self.workspace_ml_client, online_endpoint_name)