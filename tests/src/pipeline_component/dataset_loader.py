from datasets import load_dataset
import pandas as pd
from utils.logging import get_logger
import os

import sys

logger = get_logger(__name__)


class LoadDataset:
    def __init__(self, task, data_path) -> None:
        self.task = task
        self.data_path = data_path

    def translation(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "wmt16", "ro-en", split="test", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["input_string"] = test_data_df["translation"].apply(lambda x: x["en"])
                test_data_df["ro"] = test_data_df["translation"].apply(lambda x: x["ro"])
                test_data_df.to_json(self.data_path, lines=True, orient="records")
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")
