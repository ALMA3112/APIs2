"""
Wrapper pyfunc del RUN FINAL.

El modelo registrado encapsula preprocesamiento + representación + clasificador:
recibe TEXTO CRUDO y devuelve 'negative'/'positive' (Sección 5).
"""
from __future__ import annotations

import mlflow.pyfunc


class SentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib

        self.pipeline = joblib.load(context.artifacts["pipeline"])
        from text_pipeline import full_preprocess
        self._full_preprocess = full_preprocess
        self._labels = {0: "negative", 1: "positive"}

    @staticmethod
    def _to_list(model_input) -> list[str]:
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            col = "text" if "text" in model_input.columns else model_input.columns[0]
            return model_input[col].astype(str).tolist()
        if isinstance(model_input, pd.Series):
            return model_input.astype(str).tolist()
        if isinstance(model_input, str):
            return [model_input]
        return [str(x) for x in list(model_input)]

    def predict(self, context, model_input, params=None):
        texts = self._to_list(model_input)
        processed = self._full_preprocess(texts)
        preds = self.pipeline.predict(processed)
        return [self._labels[int(p)] for p in preds]
