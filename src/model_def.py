import re
import joblib
import pandas as pd
import mlflow.pyfunc

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")

def b0_prep(t):
    t = str(t).lower()
    t = _URL.sub(" url ", t)
    t = _MENTION.sub(" user ", t)
    return _WS.sub(" ", t).strip()

LABELS = {0: "negative", 1: "positive"}

class SentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        bundle = joblib.load(context.artifacts["pipeline"])
        self.vec, self.clf = bundle["vec"], bundle["clf"]

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = [str(t) for t in model_input]
        X = self.vec.transform([b0_prep(t) for t in texts])
        return [LABELS[int(y)] for y in self.clf.predict(X)]
