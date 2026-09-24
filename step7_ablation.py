"""
Ablación (Sección 4 / A.4). El candidato (C_LOGREG) difiere de B0 en DOS
decisiones:
  preprocessing.elongation  (B0: keep -> candidato: normalize)
  representation             (B0: bow[1,1] -> candidato: tfidf[1,2])
Se revierte cada una por separado, manteniendo fijo el resto del candidato,
y se compara el macro_f1_mean de cada ablación contra el del candidato.
macro_f1_delta = macro_f1_mean(candidato) - macro_f1_mean(ablación)  (A.3)

lab_ablation_parent_run_id apunta al run C_LOGREG ya registrado (no se crea
un run "CANDIDATE" nuevo, porque C_LOGREG YA ES el pipeline candidato).
"""
import copy

import mlflow
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import set_tracking, macro_f1, fold_metrics, log_json_artifact
from step3_b0 import preprocess_b0, make_config as b0_config
from step4_preprocessing import reduce_elongation

ELONGATION_SPEC = "reduce_repeated_chars_to_2"


def candidate_config() -> dict:
    cfg = copy.deepcopy(b0_config())
    cfg["preprocessing"]["elongation"] = "normalize"
    cfg["preprocessing"]["elongation_spec"] = ELONGATION_SPEC
    cfg["representation"] = {
        "type": "tfidf", "ngram_range": [1, 2],
        "library": "scikit-learn", "library_version": sklearn.__version__,
        "spacy_model": None, "spacy_model_version": None, "parameters": {},
    }
    return cfg


def ablation_config(revert: str) -> dict:
    cfg = candidate_config()
    if revert == "preprocessing.elongation":
        cfg["preprocessing"]["elongation"] = "keep"
        cfg["preprocessing"]["elongation_spec"] = None
    elif revert == "representation":
        cfg["representation"] = {
            "type": "bow", "ngram_range": [1, 1],
            "library": "scikit-learn", "library_version": sklearn.__version__,
            "spacy_model": None, "spacy_model_version": None, "parameters": {},
        }
    else:
        raise ValueError(f"decisión no reconocida: {revert}")
    return cfg


def build_vectorizer(rep_cfg: dict):
    if rep_cfg["type"] == "tfidf":
        return TfidfVectorizer(ngram_range=tuple(rep_cfg["ngram_range"]))
    return CountVectorizer(ngram_range=tuple(rep_cfg["ngram_range"]))


def run_cv(texts, labels, partitions_df, indices, rep_cfg):
    df = pd.DataFrame({"index": indices, "text_proc": texts, "label": labels})
    merged = df.merge(partitions_df, on="index", how="inner")
    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        pipe = Pipeline([
            ("vectorizer", build_vectorizer(rep_cfg)),
            ("clf", LogisticRegression()),
        ])
        pipe.fit(train_part["text_proc"], train_part["label"])
        preds = pipe.predict(val_part["text_proc"])
        s = macro_f1(val_part["label"], preds)
        scores.append(s)
        print(f"  fold {fold_id}: macro_f1={s:.4f}")
    return fold_metrics(scores)


def find_candidate_run_id(protocol_run_id: str) -> str:
    exp = mlflow.get_experiment_by_name("nlp-lab2-sentiment140")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"tags.lab_experiment_id = 'C_LOGREG' and "
            f"tags.lab_protocol_run_id = '{protocol_run_id}'"
        ),
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError("No se encontró el run C_LOGREG. Corre step6_classifier.py primero.")
    return runs.iloc[0]["run_id"], runs.iloc[0]["metrics.macro_f1_mean"]


def log_ablation_run(name, tag_experiment_id, metrics, config, protocol_run_id,
                      candidate_run_id, reverted_decision, delta, member_id="carlos.cardona02"):
    arn = "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("lab_run_type", "experiment")
        mlflow.set_tag("lab_protocol_run_id", protocol_run_id)
        mlflow.set_tag("lab_experiment_id", tag_experiment_id)
        mlflow.set_tag("lab_stage", "ablation")
        mlflow.set_tag("lab_member_id", member_id)
        mlflow.set_tag("lab_configuration_id", f"CFG_{tag_experiment_id}")
        mlflow.set_tag("notebook_arn", arn)
        mlflow.set_tag("lab_ablation_parent_run_id", candidate_run_id)
        mlflow.log_param("ablation_reverted_decision", reverted_decision)
        mlflow.log_metric("macro_f1_delta", delta)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        log_json_artifact(config, "run/configuration.json")
        log_json_artifact({"ResourceArn": arn}, "provenance/sagemaker-resource-metadata.json")
        print(f"  RUN {name} CREADO: {run.info.run_id}\n")
        return run.info.run_id


def main():
    set_tracking()
    with open(".protocol_run_id") as fh:
        protocol_run_id = fh.read().strip()

    candidate_run_id, candidate_mean = find_candidate_run_id(protocol_run_id)
    print(f"Candidato = run C_LOGREG ya registrado: {candidate_run_id}")
    print(f"macro_f1_mean del candidato: {candidate_mean:.4f}\n")

    sample_df = pd.read_parquet(".sample.parquet")
    part_path = mlflow.artifacts.download_artifacts(
        run_id=protocol_run_id, artifact_path="protocol/partitions.csv"
    )
    partitions_df = pd.read_csv(part_path)

    print("Preparando textos (crudo y con alargamientos normalizados) ...")
    base_clean = sample_df["text"].apply(preprocess_b0)
    elong_texts = base_clean.apply(reduce_elongation)

    print("=== ABLATION: revertir preprocessing.elongation ===")
    abl_cfg = ablation_config("preprocessing.elongation")
    abl_metrics = run_cv(base_clean, sample_df["label"], partitions_df, sample_df["index"], abl_cfg["representation"])
    delta = candidate_mean - abl_metrics["macro_f1_mean"]
    print(f"  macro_f1_delta = {delta:.4f}")
    log_ablation_run("ABLATION_ELONGATION", "ABLATION", abl_metrics, abl_cfg, protocol_run_id,
                      candidate_run_id, "preprocessing.elongation", delta)

    print("=== ABLATION: revertir representation ===")
    abl_cfg2 = ablation_config("representation")
    abl_metrics2 = run_cv(elong_texts, sample_df["label"], partitions_df, sample_df["index"], abl_cfg2["representation"])
    delta2 = candidate_mean - abl_metrics2["macro_f1_mean"]
    print(f"  macro_f1_delta = {delta2:.4f}")
    log_ablation_run("ABLATION_REPRESENTATION", "ABLATION", abl_metrics2, abl_cfg2, protocol_run_id,
                      candidate_run_id, "representation", delta2)

    print("\n=== RESUMEN DE ABLACIÓN ===")
    print(f"Candidato (C_LOGREG):            macro_f1_mean={candidate_mean:.4f}  run_id={candidate_run_id}")
    print(f"Sin normalizar alargamientos:     macro_f1_mean={abl_metrics['macro_f1_mean']:.4f}  delta={delta:.4f}")
    print(f"Con BoW (revert representation):  macro_f1_mean={abl_metrics2['macro_f1_mean']:.4f}  delta={delta2:.4f}")
    print("\nUn delta positivo significa que esa decisión SÍ ayuda (el candidato es mejor que revertirla).")
    with open(".candidate_run_id", "w") as fh:
        fh.write(candidate_run_id)


if __name__ == "__main__":
    main()
