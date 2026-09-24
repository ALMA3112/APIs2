"""
Etapa "Representación" (Sección 3 / A.4). Con el preprocesamiento seleccionado
(P_ELONGATION: el único que superó a B0 en la etapa anterior) compara:
  R_BOW           BoW unigramas
  R_TFIDF_UNI     TF-IDF unigramas
  R_TFIDF_UNI_BI  TF-IDF unigramas+bigramas
  R_SPACY         embeddings de spaCy con vectores preentrenados
Condición fija: preprocesamiento seleccionado + Logistic Regression.
"""
import copy

import mlflow
import numpy as np
import pandas as pd
import sklearn
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import set_tracking, macro_f1, fold_metrics, log_json_artifact
from step3_b0 import preprocess_b0, make_config as b0_config
from step4_preprocessing import reduce_elongation

ELONGATION_SPEC = "reduce_repeated_chars_to_2"

SPACY_VECTOR_MODEL = "en_core_web_md"
DOCUMENT_VECTOR_METHOD = "spacy_doc_vector_mean"

_nlp_vectors = None


def get_vector_nlp():
    global _nlp_vectors
    if _nlp_vectors is None:
        _nlp_vectors = spacy.load(SPACY_VECTOR_MODEL)
    return _nlp_vectors


def doc_vectors(texts: list[str]) -> np.ndarray:
    nlp_v = get_vector_nlp()
    vecs = [doc.vector for doc in nlp_v.pipe(texts, batch_size=200)]
    return np.vstack(vecs)


def selected_preprocessing_config() -> dict:
    """Preprocesamiento seleccionado tras la Sección 3: P_ELONGATION."""
    cfg = copy.deepcopy(b0_config())
    cfg["preprocessing"]["elongation"] = "normalize"
    cfg["preprocessing"]["elongation_spec"] = ELONGATION_SPEC
    return cfg


def representation_config(rep_type: str, ngram_range=None) -> dict:
    cfg = selected_preprocessing_config()
    rep = {
        "type": rep_type,
        "ngram_range": ngram_range if ngram_range is not None else [],
        "library": "scikit-learn" if rep_type in ("bow", "tfidf") else "spacy",
        "library_version": sklearn.__version__ if rep_type in ("bow", "tfidf") else spacy.__version__,
        "spacy_model": None,
        "spacy_model_version": None,
        "parameters": {},
    }
    if rep_type == "spacy_embedding":
        rep["spacy_model"] = SPACY_VECTOR_MODEL
        rep["spacy_model_version"] = get_vector_nlp().meta.get("version", "unknown")
        rep["parameters"] = {"document_vector_method": DOCUMENT_VECTOR_METHOD}
    cfg["representation"] = rep
    return cfg


def _log_run(name, tag_experiment_id, metrics, config, protocol_run_id, member_id="carlos.cardona02"):
    arn = "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("lab_run_type", "experiment")
        mlflow.set_tag("lab_protocol_run_id", protocol_run_id)
        mlflow.set_tag("lab_experiment_id", tag_experiment_id)
        mlflow.set_tag("lab_stage", "representation")
        mlflow.set_tag("lab_member_id", member_id)
        mlflow.set_tag("lab_configuration_id", f"CFG_{tag_experiment_id}")
        mlflow.set_tag("notebook_arn", arn)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        log_json_artifact(config, "run/configuration.json")
        log_json_artifact({"ResourceArn": arn}, "provenance/sagemaker-resource-metadata.json")
        print(f"  RUN {name} CREADO: {run.info.run_id}\n")
        return run.info.run_id


def run_text_representation(name, tag_experiment_id, vectorizer_factory, texts, labels,
                             partitions_df, indices, config, protocol_run_id):
    df = pd.DataFrame({"index": indices, "text_proc": texts, "label": labels})
    merged = df.merge(partitions_df, on="index", how="inner")

    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        pipe = Pipeline([
            ("vectorizer", vectorizer_factory()),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(train_part["text_proc"], train_part["label"])
        preds = pipe.predict(val_part["text_proc"])
        s = macro_f1(val_part["label"], preds)
        scores.append(s)
        print(f"  fold {fold_id}: macro_f1={s:.4f}")

    metrics = fold_metrics(scores)
    print(f"  Resumen {name}:", metrics)
    run_id = _log_run(name, tag_experiment_id, metrics, config, protocol_run_id)
    return run_id, metrics


def run_precomputed_representation(name, tag_experiment_id, X, labels, partitions_df,
                                    indices, config, protocol_run_id):
    df = pd.DataFrame({"index": indices, "label": labels})
    merged = df.merge(partitions_df, on="index", how="inner")
    pos = {idx: i for i, idx in enumerate(indices)}
    order = merged["index"].map(pos).to_numpy()
    X_ordered = X[order]

    scores = []
    for fold_id in range(3):
        train_mask = (merged["fold"] != fold_id).to_numpy()
        val_mask = (merged["fold"] == fold_id).to_numpy()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_ordered[train_mask], merged.loc[train_mask, "label"])
        preds = clf.predict(X_ordered[val_mask])
        s = macro_f1(merged.loc[val_mask, "label"], preds)
        scores.append(s)
        print(f"  fold {fold_id}: macro_f1={s:.4f}")

    metrics = fold_metrics(scores)
    print(f"  Resumen {name}:", metrics)
    run_id = _log_run(name, tag_experiment_id, metrics, config, protocol_run_id)
    return run_id, metrics


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

    print("=== R_BOW ===")
    cfg = representation_config("bow", ngram_range=[1, 1])
    results["R_BOW"] = run_text_representation(
        "R_BOW", "R_BOW", lambda: CountVectorizer(ngram_range=(1, 1)),
        elong_texts, sample_df["label"], partitions_df, sample_df["index"], cfg, protocol_run_id,
    )

    print("=== R_TFIDF_UNI ===")
    cfg = representation_config("tfidf", ngram_range=[1, 1])
    results["R_TFIDF_UNI"] = run_text_representation(
        "R_TFIDF_UNI", "R_TFIDF_UNI", lambda: TfidfVectorizer(ngram_range=(1, 1)),
        elong_texts, sample_df["label"], partitions_df, sample_df["index"], cfg, protocol_run_id,
    )

    print("=== R_TFIDF_UNI_BI ===")
    cfg = representation_config("tfidf", ngram_range=[1, 2])
    results["R_TFIDF_UNI_BI"] = run_text_representation(
        "R_TFIDF_UNI_BI", "R_TFIDF_UNI_BI", lambda: TfidfVectorizer(ngram_range=(1, 2)),
        elong_texts, sample_df["label"], partitions_df, sample_df["index"], cfg, protocol_run_id,
    )

    print("=== R_SPACY ===")
    print(f"Calculando embeddings con {SPACY_VECTOR_MODEL} (puede tardar) ...")
    X = doc_vectors(elong_texts.tolist())
    cfg = representation_config("spacy_embedding")
    results["R_SPACY"] = run_precomputed_representation(
        "R_SPACY", "R_SPACY", X, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("\n=== RESUMEN DE LA ETAPA DE REPRESENTACIÓN ===")
    for name, (run_id, metrics) in results.items():
        print(f"{name:16s} macro_f1_mean={metrics['macro_f1_mean']:.4f}  run_id={run_id}")
    print("\nElige la representación con mejor macro_f1_mean para pasar a la etapa de Clasificador.")


if __name__ == "__main__":
    main()
