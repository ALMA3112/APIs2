import re, os, pandas as pd, mlflow
from mlflow.tracking import MlflowClient
from datasets import load_dataset

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://34.228.215.204:5000")
EXPERIMENT = "nlp-lab2-sentiment140"
DATASET = "adilbekovich/Sentiment140Twitter"
REV = "b6037e127257d95b9b23d31f78b264b9ebe697fd"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")

def b0_prep(t):
    t = str(t).lower()
    t = _URL.sub(" url ", t)
    t = _MENTION.sub(" user ", t)
    return _WS.sub(" ", t).strip()

def client():
    mlflow.set_tracking_uri(URI)
    return MlflowClient()

def protocol_run_id(c):
    exp = c.get_experiment_by_name(EXPERIMENT)
    runs = c.search_runs([exp.experiment_id], filter_string="tags.lab_run_type = 'protocol'")
    assert len(runs) == 1, f"protocolos: {len(runs)}"
    return runs[0].info.run_id

def load_train():
    tr = load_dataset(DATASET, revision=REV, split="train").to_pandas()
    assert len(tr) == 1_360_000, len(tr)
    return tr

def load_partitions(c=None):
    c = c or client()
    path = mlflow.artifacts.download_artifacts(run_id=protocol_run_id(c),
                                               artifact_path="protocol/partitions.csv",
                                               dst_path=os.path.expanduser("~/SageMaker/lab2/protocol"))
    return pd.read_csv(path)

def sample_with_folds(train, parts):
    s = train.iloc[parts["index"].values].copy()
    s["orig_index"] = parts["index"].values
    s["fold"] = parts["fold"].values
    return s
