"""
T0: referencia trivial (Sección 2). En cada fold predice la clase más
frecuente del entrenamiento; en empate, predice negative.
Se registra como run experimental con lab_experiment_id=T0 (A.4).
"""
import json
import mlflow
import pandas as pd
from common import set_tracking, macro_f1, fold_metrics, log_json_artifact

T0_CONFIG = {
    "preprocessing": None,
    "representation": None,
    "classifier": {
        "type": "most_frequent",
        "library": "custom",
        "library_version": "1.0",
        "parameters": {},
    },
}


def most_frequent_predict(y_train, n):
    counts = y_train.value_counts()
    if len(counts) >= 2 and counts.iloc[0] == counts.iloc[1]:
        majority = 0  # empate -> negative (0)
    else:
        majority = counts.idxmax()
    return [majority] * n


def main():
    set_tracking()
    with open(".protocol_run_id") as fh:
        protocol_run_id = fh.read().strip()

    sample_df = pd.read_parquet(".sample.parquet")
    partitions_df = pd.read_csv(".protocol_partitions_cache.csv") if False else None
    # Releemos las particiones directamente de MLflow para no duplicar lógica:
    import mlflow.artifacts as artifacts
    part_path = mlflow.artifacts.download_artifacts(
        run_id=protocol_run_id, artifact_path="protocol/partitions.csv"
    )
    partitions_df = pd.read_csv(part_path)

    merged = sample_df.merge(partitions_df, on="index", how="inner")
    assert len(merged) == len(sample_df), "la muestra y las particiones no calzan"

    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        preds = most_frequent_predict(train_part["label"], len(val_part))
        scores.append(macro_f1(val_part["label"], preds))
        print(f"fold {fold_id}: macro_f1={scores[-1]:.4f}")

    metrics = fold_metrics(scores)
    print("Resumen T0:", metrics)

    with mlflow.start_run(run_name="T0") as run:
        mlflow.set_tag("lab_run_type", "experiment")
        mlflow.set_tag("lab_protocol_run_id", protocol_run_id)
        mlflow.set_tag("lab_experiment_id", "T0")
        mlflow.set_tag("lab_stage", "reference")
        mlflow.set_tag("lab_member_id", "carlos.cardona02")
        mlflow.set_tag("lab_configuration_id", "CFG_T0")
        mlflow.set_tag("notebook_arn", "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0")

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        log_json_artifact(T0_CONFIG, "run/configuration.json")
        # Procedencia simulada para el ensayo local (en AWS será el archivo real)
        log_json_artifact(
            {"ResourceArn": "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"},
            "provenance/sagemaker-resource-metadata.json",
        )
        print(f"\nRUN T0 CREADO: {run.info.run_id}")


if __name__ == "__main__":
    main()
