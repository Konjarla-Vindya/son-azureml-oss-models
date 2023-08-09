")
# directory to write logs
parser.add_argument("--log_dir", type=str, default="../logs")

args = parser.parse_args()
parallel_tests = int(args.parallel_tests)

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print ("::error Auth failed, DefaultAzureCredential not working: \n{e}")
    exit (1)
# Connect to the HuggingFaceHub registry
registry_ml_client = MLClient(credential, registry_name="HuggingFace")

queue = []

# move this to config file later
templates=['transformers-cpu-small', 'transformers-cpu-medium', 'transformers-cpu-large','transformers-cpu-extra-large', 'transformers-gpu-medium']


def load_workspace_config():
    with open(args.workspace_list) as f:
        return json.load(f)
    
# function to assign models to queues
# assign each model from models to a thread per workspace in a round robin fashion by appending to a list called 'models' in the queue dictionary
def assign_models_to_queues(models, workspace_list):
  queue = {}
  i=0
  while i < len(models):
      for workspace in workspace_list:
          print (f"workspace instance: {workspace}")
          for thread in range(parallel_tests):
              print (f"thread instance: {thread}")
              if i < len(models):
                  if workspace not in queue:
                      queue[workspace] = {}
                      print("queue[workspace]",queue[workspace])
                  if thread not in queue[workspace]:
                      queue[workspace][thread] = []
                  queue[workspace][thread].append(models[i])
                  print("queue[workspace][thread]",queue[workspace][thread])
                  i=i+1
def main():
    # get list of models from registry
    if args.mode == "registry":
        models = get_model_containers(args.registry_name)
    elif args.mode == "file":
        models = load_model_list_file(args.model_list_file)
    else:
        print (f"::error Invalid mode {args.mode}")
        exit (1)
    print (f"Found {len(models)} models")
    # load workspace_list_json
    workspace_list = load_workspace_config()
    print (f"Found {len(workspace_list)} workspaces")
    # assign models to queues
    queue = assign_models_to_queues(models, workspace_list)
    print (f"Created queues")
    # create queue files
    create_queue_files(queue, workspace_list)
    print (f"Created queue files")
    # create workflow files
    create_workflow_files(queue, workspace_list)
    print (f"Created workflow files")
    print (f"Summary:")
    print (f"  Models: {len(models)}")
    print (f"  Workspaces: {len(workspace_list)}")
    print (f"  Parallel tests: {parallel_tests}")
    print (f"  Total queues: {len(workspace_list)*parallel_tests}")
    print (f"  Average models per queue: {int(len(models)/(len(workspace_list)*parallel_tests))}")

        
if __name__ == "__main__":
    main()
