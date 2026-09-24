"""
Crea el ÚNICO run de protocolo (Sección 2 / A.2): fija dataset, muestra y folds.
No entrena ningún modelo. Genera protocol/partitions.csv y protocol/members.csv.
"""
import os
import mlflow
from common import (
    set_tracking, load_train_test, build_sample_and_folds,
    DATASET_ID, DATASET_REVISION, RANDOM_SEED, CV_FOLDS, SAMPLE_SIZE,
)

# --- Integrantes del equipo: EDITA ESTA LISTA con tus datos reales ---
# En AWS será el ARN real del Notebook Instance asignado por el curso.
MEMBERS = [
    {"member_id": "carlos.cardona02", "notebook_arn": "arn:aws:sagemaker:us-east-1:868565605844:notebook-instance/MLFLOW0"},
]


def main():
    set_tracking()
    print(f"Cargando dataset {DATASET_ID}@{DATASET_REVISION} ...")
    train_df, _test_df = load_train_test()
    print(f"train: {len(train_df)} filas")

    print(f"Construyendo muestra de {SAMPLE_SIZE} y {CV_FOLDS} folds (semilla {RANDOM_SEED}) ...")
    sample_df, partitions_df = build_sample_and_folds(train_df)
    print(f"Muestra: {len(sample_df)} filas. Partitions: {len(partitions_df)} filas.")

    with mlflow.start_run(run_name="protocol") as run:
        mlflow.set_tag("lab_run_type", "protocol")

        mlflow.log_param("dataset_id", DATASET_ID)
        mlflow.log_param("dataset_revision", DATASET_REVISION)
        mlflow.log_param("sampling_strategy", "stratified")
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_shuffle", True)

        # protocol/partitions.csv
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            part_path = os.path.join(tmp, "partitions.csv")
            partitions_df.to_csv(part_path, index=False, encoding="utf-8")
            mlflow.log_artifact(part_path, artifact_path="protocol")

            members_path = os.path.join(tmp, "members.csv")
            import csv
            with open(members_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["member_id", "notebook_arn"])
                for m in sorted(MEMBERS, key=lambda x: x["member_id"]):
                    w.writerow([m["member_id"], m["notebook_arn"]])
            mlflow.log_artifact(members_path, artifact_path="protocol")

        protocol_run_id = run.info.run_id
        print(f"\nRUN DE PROTOCOLO CREADO: {protocol_run_id}")
        print("Guarda este run_id, lo necesitas para los siguientes scripts.")

        # lo guardamos en un archivo local para que los próximos scripts lo lean solos
        with open(".protocol_run_id", "w") as fh:
            fh.write(protocol_run_id)

        # también dejamos la muestra en disco para no volver a muestrear cada vez
        sample_df.to_parquet(".sample.parquet")
        _test_df.to_parquet(".test.parquet")


if __name__ == "__main__":
    main()
