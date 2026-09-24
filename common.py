"""
Utilidades compartidas por todos los scripts de entrenamiento.

IMPORTANTE: SAMPLE_SIZE se controla con la variable de entorno LAB_SAMPLE_SIZE.
  - En tu PC (ensayo): la dejamos pequeña (p.ej. 2000) para que todo corra en segundos.
  - En SageMaker (entrega real): se pone en 200000, tal como exige la guía.
La LÓGICA es idéntica en ambos casos; solo cambia el tamaño.
"""
import os
import json
import numpy as np
import pandas as pd
import mlflow
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# --- Parámetros fijos por la guía (Sección 2) ---
DATASET_ID = "adilbekovich/Sentiment140Twitter"
DATASET_REVISION = "b6037e127257d95b9b23d31f78b264b9ebe697fd"
RANDOM_SEED = 42
CV_FOLDS = 3

# --- Parámetro ajustable: tamaño de muestra ---
SAMPLE_SIZE = int(os.getenv("LAB_SAMPLE_SIZE", "2000"))  # 200000 en la entrega real

EXPERIMENT_NAME = "nlp-lab2-sentiment140"

LABEL_MAP = {0: "negative", 1: "positive"}


def set_tracking():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def load_train_test():
    """Carga train/test del dataset, fijando la revisión exacta (Sección 2)."""
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    return train_df, test_df


def build_sample_and_folds(train_df: pd.DataFrame):
    """
    Muestra estratificada de SAMPLE_SIZE registros (semilla 42), conservando el
    índice original de train, y 3 folds estratificados (semilla 42).
    Devuelve (sample_df, partitions_df) donde partitions_df tiene columnas
    index,fold exactamente como exige A.2.
    """
    train_df = train_df.reset_index().rename(columns={"index": "orig_index"})
    # si el dataset no trae 'label' con ese nombre exacto, ajustar aquí
    label_col = "label" if "label" in train_df.columns else "sentiment"

    frac = SAMPLE_SIZE / len(train_df)
    sample_df = (
        train_df.groupby(label_col, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_SEED))
    )
    # Ajuste fino para caer exacto en SAMPLE_SIZE (por redondeos del frac)
    if len(sample_df) > SAMPLE_SIZE:
        sample_df = sample_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    sample_df = sample_df.sort_values("orig_index").reset_index(drop=True)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_of = np.empty(len(sample_df), dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(sample_df, sample_df[label_col])):
        fold_of[val_idx] = fold_id

    partitions_df = pd.DataFrame({
        "index": sample_df["orig_index"].values,
        "fold": fold_of,
    }).sort_values("index").reset_index(drop=True)

    sample_df = sample_df.rename(columns={label_col: "label", "orig_index": "index"})
    return sample_df[["index", "text", "label"]], partitions_df


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def fold_metrics(scores: list[float]) -> dict:
    """mean y std POBLACIONAL (ddof=0), como exige A.2."""
    arr = np.array(scores, dtype=float)
    return {
        "macro_f1_fold_0": float(arr[0]),
        "macro_f1_fold_1": float(arr[1]),
        "macro_f1_fold_2": float(arr[2]),
        "macro_f1_mean": float(arr.mean()),
        "macro_f1_std": float(arr.std(ddof=0)),
    }


def log_json_artifact(obj: dict, artifact_path: str):
    """Guarda un dict como JSON UTF-8 y lo sube como artefacto en artifact_path."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.basename(artifact_path)
        subdir = os.path.dirname(artifact_path)
        local = os.path.join(tmp, fname)
        with open(local, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        mlflow.log_artifact(local, artifact_path=subdir or None)
