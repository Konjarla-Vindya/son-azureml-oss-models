from azure.ai.ml.dsl import pipeline
class Pipeline:
    def __init__(self, import_model ) -> None :
        self.import_model  = import_model

    @pipeline
    def create_pipeline(self, model_id, compute) -> dict :
        import_model_job = self.import_model(model_id=model_id, compute=compute)
        # Set job to not continue on failure
        import_model_job.settings.continue_on_step_failure = False 

        return {
        "model_registration_details": import_model_job.outputs.model_registration_details
       }