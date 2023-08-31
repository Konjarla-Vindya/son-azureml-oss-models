import json
from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget

# Load configurations from JSON file
with open('azure_multi_compute.json', 'r') as f:
    configs = json.load(f)

for config in configs:
    # Initialize workspace
    try:
        ws = Workspace(
            subscription_id=config['subscription_id'],
            resource_group=config['resource_group'],
            workspace_name=config['workspace_name']
        )
    except Exception as e:
        print(f"An error occurred while initializing the workspace: {e}")
        continue

    # Create compute clusters
    for compute in config['computes']:
        try:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=compute['vm_size'],
                max_nodes=compute['max_nodes']
            )
            compute_target = ComputeTarget.create(
                ws,
                compute['compute_name'],
                compute_config
            )
            compute_target.wait_for_completion(show_output=True)
        except Exception as e:
            print(f"An error occurred while creating the compute cluster: {e}")
