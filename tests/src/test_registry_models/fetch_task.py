import re
from huggingface_hub import HfApi
import pandas as pd
from utils.logging import get_logger
LIST_OF_COLUMNS = ['modelId', 'downloads',
                   'lastModified', 'tags', 'pipeline_tag']
TASK_NAME = ['fill-mask', 'token-classification', 'question-answering',
             'summarization', 'text-generation', 'text-classification', 'translation']
STRING_TO_CHECK = 'transformers'
logger = get_logger(__name__)


class HfTask:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_task(self):
        hf_api = HfApi()
        logger.info(
            "Fetching all data from the transformer sorted based on the last modified date")
        # Get all the1 models in the list
        models = hf_api.list_models(
            full=True, sort='lastModified', direction=-1)
        # Unpack all values from the generator object
        required_data = [i for i in models]

        daata_dict = {}
        # Loop through the list
        for data in required_data:
            # Loop through all the column present in the list
            for key in data.__dict__.keys():
                if key in LIST_OF_COLUMNS:
                    # Check the dictionary already contains a value for that particular column
                    if daata_dict.get(key) is None:
                        # If the column and its value is not present then insert column and an empty list pair to the dictionary
                        daata_dict[key] = []
                    # Get the value for that particular column
                    values = daata_dict.get(key)
                    if key == 'tags':
                        # If its tag column extract value if it is nonne then bydefault return a list with string Empty
                        values.append(data.__dict__.get(key, ["Empty"]))
                    else:
                        values.append(data.__dict__.get(key, "Empty"))
                    daata_dict[key] = values
        # Convert dictionary to the dataframe
        df = pd.DataFrame(daata_dict)
        # Find the data with the model which will be having trasnfomer tag
        df = df[df.tags.apply(lambda x: STRING_TO_CHECK in x)]
        # Retrive the data whose task is in the list
        df = df[df['pipeline_tag'].isin(TASK_NAME)]
        # Find the data with that particular name
        required_data = df[df.modelId.apply(lambda x: x == self.model_name)]
        # Get the task
        required_data = required_data["pipeline_tag"].to_string()
        # Create pattern to avoid number and space
        pattern = r'[0-9\s+]'
        # Replace number and space
        final_data = re.sub(pattern, '', required_data)
        logger.info(f"The specified task is this one : {final_data}")
        return final_data
