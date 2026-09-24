"""
Etapa "Clasificador" (Sección 3 / A.4). Con la representación seleccionada
(TF-IDF unigramas+bigramas, sobre preprocesamiento P_ELONGATION) compara:
  C_LOGREG      Logistic Regression
  C_LINEAR_SVM  Linear SVM
  C_SGD         SGDClassifier
Condición fija: datos, folds, preprocesamiento y representación seleccionados.
"""
import copy

import mlflow
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import set_tracking, macro_f1, fold_metrics, log_json_artifact
from step3_b0 import preprocess_b0, make_config as b0_config
from step4_preprocessing import reduce_elongation

ELONGATION_SPEC = "reduce_repeated_chars_to_2"


def selected_config(classifier_type: str) -> dict:
    cfg = copy.deepcopy(b0_config())
    cfg["preprocessing"]["elongation"] = "normalize"
    cfg["preprocessing"]["elongation_spec"] = ELONGATION_SPEC
    cfg["representation"] = {
        "type": "tfidf", "ngram_range": [1, 2],
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "spacy_model": None, "spacy_model_version": None, "parameters": {},
    }
    cfg["classifier"] = {
        "type": classifier_type,
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "parameters": {},
    }
    return cfg


CLASSIFIERS = {
    "C_LOGREG": ("logistic_regression", lambda: LogisticRegression()),
    "C_LINEAR_SVM": ("linear_svm", lambda: LinearSVC()),
    "C_SGD": ("sgd", lambda: SGDClassifier()),
}


def run_classifier_experiment(name, tag_experiment_id, clf_factory, texts, labels,
                               partitions_df, indices, config, protocol_run_id, member_id="carlos.cardona02"):
    df = pd.DataFrame({"index": indices, "text_proc": texts, "label": labels})
    merged = df.merge(partitions_df, on="index", how="inner")

    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        pipe = Pipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", clf_factory()),
        ])
        pipe.fit(train_part["text_proc"], train_part["label"])
        preds = pipe.predict(val_part["text_proc"])
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
        mlflow.set_tag("lab_stage", "classifier")
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

    print("Aplicando preprocesamiento seleccionado (P_ELONGATION) ...")
    base_clean = sample_df["text"].apply(preprocess_b0)
    elong_texts = base_clean.apply(reduce_elongation)

    results = {}
    for tag_id, (ctype, factory) in CLASSIFIERS.items():
        print(f"=== {tag_id} ===")
        cfg = selected_config(ctype)
        results[tag_id] = run_classifier_experiment(
            tag_id, tag_id, factory, elong_texts, sample_df["label"],
            partitions_df, sample_df["index"], cfg, protocol_run_id,
        )

    print("\n=== RESUMEN DE LA ETAPA DE CLASIFICADOR ===")
    for name, (run_id, metrics) in results.items():
        print(f"{name:14s} macro_f1_mean={metrics['macro_f1_mean']:.4f}  run_id={run_id}")
    print("\nEsta es tu decisión de PIPELINE CANDIDATO: preprocesamiento + representación")
    print("+ clasificador con mejor macro_f1_mean. Sigue con la ablación (A.4).")


if __name__ == "__main__":
    main()
