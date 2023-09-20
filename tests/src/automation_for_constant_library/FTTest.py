import os
from FT_initial_automation import load_model  
import mlflow
import transformers
def data_set():
    exit_status = os.system("python ./download-dataset.py --download_dir emotion-dataset")
    print("exit_status----------",exit_status)
    if exit_status != 0:
        raise Exception("Error downloading dataset")
    # load the ./emotion-dataset/train.jsonl file into a pandas dataframe and show the first 5 rows
    

    pd.set_option(
        "display.max_colwidth", 0
    )  # set the max column width to 0 to display the full text
    df = pd.read_json("./emotion-dataset/train.jsonl", lines=True)
    df.head()

    # load the id2label json element of the ./emotion-dataset/label.json file into pandas table with keys as 'label' column of int64 type and values as 'label_string' column as string type
    

    with open("./emotion-dataset/label.json") as f:
        id2label = json.load(f)
        id2label = id2label["id2label"]
        label_df = pd.DataFrame.from_dict(
            id2label, orient="index", columns=["label_string"]
        )
        label_df["label"] = label_df.index.astype("int64")
        label_df = label_df[["label", "label_string"]]
        label_df.head()
    print("downloaded data set-------------")
if __name__ == "__main__":
  model_source_uri=os.environ.get('model_source_uri')
  test_model_name = os.environ.get('test_model_name')
  print("test_model_name-----------------",test_model_name)
  loaded_model = mlflow.transformers.load_model(model_uri=model_source_uri, return_type="pipeline")
  print("loaded_model---------------------",loaded_model)
  data_set()
