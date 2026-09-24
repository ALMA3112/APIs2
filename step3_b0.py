"""
B0: baseline (Sección 2 / A.3). minúsculas + url/mention -> token + espacios
normalizados. BoW unigramas + Logistic Regression, hiperparámetros por defecto.
No tocar hiperparámetros aunque salga warning de convergencia (así lo pide la guía).
"""
import re
import warnings
import sklearn
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from common import set_tracking, macro_f1, fold_metrics, log_json_artifact

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def preprocess_b0(text: str) -> str:
    t = text.lower()
    t = URL_RE.sub(" url ", t)
    t = MENTION_RE.sub(" user ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def make_config():
    return {
        "preprocessing": {
            "lowercase": True, "url": "token:url", "mention": "token:user",
            "whitespace": "normalize", "stopwords": "keep", "negators": [],
            "lemmatize": False, "elongation": "keep", "elongation_spec": None,
            "emoji": "keep", "emoji_spec": None, "resources": {}, "additional": {},
        },
        "representation": {
            "type": "bow", "ngram_range": [1, 1],
            "library": "scikit-learn", "library_version": sklearn.__version__,
            "spacy_model": None, "spacy_model_version": None, "parameters": {},
        },
        "classifier": {
            "type": "logistic_regression",
            "library": "scikit-learn", "library_version": sklearn.__version__,
            "parameters": {},
        },
    }


def build_pipeline():
    return Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=(1, 1), preprocessor=preprocess_b0)),
        ("clf", LogisticRegression()),
    ])


def run_experiment(name, tag_experiment_id, tag_stage, config, build_pipeline_fn,
                    sample_df, partitions_df, protocol_run_id, member_id="carlos.cardona02"):
    merged = sample_df.merge(partitions_df, on="index", how="inner")
    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        pipe = build_pipeline_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no se ajustan hiperparámetros por un warning
            pipe.fit(train_part["text"], train_part["label"])
        preds = pipe.predict(val_part["text"])
        s = macro_f1(val_part["label"], preds)
        scores.append(s)
        print(f"  fold {fold_id}: macro_f1={s:.4f}")

    metrics = fold_metrics(scores)
    print(f"  Resumen {name}:", metrics)

    arn = "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("lab_run_type", "experiment")
        mlflow.set_tag("lab_protocol_run_id", protocol_run_id)
        mlflow.set_tag("lab_experiment_id", tag_experiment_id)
        mlflow.set_tag("lab_stage", tag_stage)
        mlflow.set_tag("lab_member_id", member_id)
        mlflow.set_tag("lab_configuration_id", f"CFG_{tag_experiment_id}")
        mlflow.set_tag("notebook_arn", arn)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        log_json_artifact(config, "run/configuration.json")
        log_json_artifact({"ResourceArn": arn}, "provenance/sagemaker-resource-metadata.json")
        print(f"  RUN {name} CREADO: {run.info.run_id}\n")
        return run.info.run_id, metrics


def main():
    set_tracking()
    with open(".protocol_run_id") as fh:
        protocol_run_id = fh.read().strip()
    sample_df = pd.read_parquet(".sample.parquet")
    part_path = mlflow.artifacts.download_artifacts(
        run_id=protocol_run_id, artifact_path="protocol/partitions.csv"
    )
    partitions_df = pd.read_csv(part_path)

    print("Entrenando B0 (BoW unigramas + Logistic Regression) ...")
    run_id, metrics = run_experiment(
        "B0", "B0", "baseline", make_config(), build_pipeline,
        sample_df, partitions_df, protocol_run_id,
    )
    with open(".b0_run_id", "w") as fh:
        fh.write(run_id)


if __name__ == "__main__":
    main()
