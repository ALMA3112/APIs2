"""
Modelo final (Sección 5). Reentrena la configuración seleccionada con TODO
train (1.360.000), evalúa UNA sola vez sobre test (240.000), y registra el
modelo en MLflow Model Registry como sentiment140@champion.

Configuración final: normalizar alargamientos + TF-IDF unigramas+bigramas +
Logistic Regression (el pipeline candidato confirmado por la ablación).

IMPORTANTE: no ajustar la configuración a partir del resultado de test.
"""
import copy
import os
import sys
import time

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "serving"))
from text_pipeline import full_preprocess  # noqa: E402

from common import set_tracking, macro_f1, log_json_artifact, DATASET_ID, DATASET_REVISION
from step3_b0 import make_config as b0_config

ELONGATION_SPEC = "reduce_repeated_chars_to_2"


def final_config() -> dict:
    cfg = copy.deepcopy(b0_config())
    cfg["preprocessing"]["elongation"] = "normalize"
    cfg["preprocessing"]["elongation_spec"] = ELONGATION_SPEC
    cfg["representation"] = {
        "type": "tfidf", "ngram_range": [1, 2],
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "spacy_model": None, "spacy_model_version": None, "parameters": {},
    }
    return cfg


def main():
    set_tracking()
    with open(".protocol_run_id") as fh:
        protocol_run_id = fh.read().strip()
    with open(".candidate_run_id") as fh:
        selected_experiment_run_id = fh.read().strip()

    print(f"Cargando dataset completo {DATASET_ID}@{DATASET_REVISION} ...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    label_col = "label" if "label" in train_df.columns else "sentiment"
    print(f"train: {len(train_df)} filas   test: {len(test_df)} filas")

    print("Preprocesando TRAIN (limpieza base + normalizar alargamientos) ...")
    t0 = time.time()
    train_texts = full_preprocess(train_df["text"].tolist())
    print(f"  listo en {time.time() - t0:.1f}s")

    print("Entrenando TF-IDF unigramas+bigramas + Logistic Regression sobre TODO train ...")
    t0 = time.time()
    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(train_texts, train_df[label_col])
    print(f"  entrenado en {time.time() - t0:.1f}s")

    print("Preprocesando TEST ...")
    t0 = time.time()
    test_texts = full_preprocess(test_df["text"].tolist())
    print(f"  listo en {time.time() - t0:.1f}s")

    print("Evaluando UNA sola vez sobre test ...")
    preds = pipe.predict(test_texts)
    test_f1 = macro_f1(test_df[label_col], preds)
    print(f"TEST macro_f1 = {test_f1:.4f}")

    joblib.dump(pipe, "pipeline.joblib")
    cfg = final_config()

    arn = "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"
    with mlflow.start_run(run_name="FINAL") as run:
        mlflow.set_tag("lab_run_type", "final")
        mlflow.set_tag("lab_protocol_run_id", protocol_run_id)
        mlflow.set_tag("lab_selected_experiment_run_id", selected_experiment_run_id)
        mlflow.set_tag("lab_configuration_id", "CFG_C_LOGREG")
        mlflow.set_tag("lab_member_id", "carlos.cardona02")
        mlflow.set_tag("notebook_arn", arn)

        mlflow.log_param("training_size", 1360000)
        mlflow.log_metric("test_macro_f1", test_f1)

        log_json_artifact(cfg, "run/configuration.json")
        log_json_artifact({"ResourceArn": arn}, "provenance/sagemaker-resource-metadata.json")

        from model_wrapper import SentimentModel

        serving_dir = os.path.join(os.path.dirname(__file__), "serving")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentModel(),
            artifacts={"pipeline": "pipeline.joblib"},
            code_paths=[
                os.path.join(serving_dir, "text_pipeline.py"),
                os.path.join(serving_dir, "model_wrapper.py"),
            ],
            pip_requirements=[
                f"scikit-learn=={sklearn.__version__}", "pandas", "joblib", "mlflow",
            ],
            registered_model_name="sentiment140",
            input_example=pd.DataFrame({"text": ["i love this"]}),
        )
        print(f"\nRUN FINAL CREADO: {run.info.run_id}")
        final_run_id = run.info.run_id

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='sentiment140'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias("sentiment140", "champion", latest.version)
    print(f"Alias 'champion' -> versión {latest.version} (run {latest.run_id})")
    print("\n¡Listo! El modelo final está registrado como sentiment140@champion.")

    with open(".final_run_id", "w") as fh:
        fh.write(final_run_id)


if __name__ == "__main__":
    main()
