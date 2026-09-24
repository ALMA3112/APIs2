"""
Etapa "Preprocesamiento" (Sección 3 / A.4). Frente a B0 evalúa por separado:
  P_STOPWORDS            eliminar stopwords de spaCy
  P_STOPWORDS_NEGATION   eliminar stopwords preservando negadores
  P_LEMMA                lematizar
  P_ELONGATION           normalizar alargamientos ("sooo" -> "soo")
  P_EMOJI                convertir emojis/emoticonos a texto
Condición fija: BoW unigramas + Logistic Regression (igual a B0 en todo lo demás).
"""
import copy
import re

import emoji
import mlflow
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import set_tracking, macro_f1, fold_metrics, log_json_artifact
from step3_b0 import preprocess_b0, make_config as b0_config

NEGATORS = ["no", "not", "n't", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "cannot"]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

ELONGATION_RE = re.compile(r"(.)\1{2,}")


def reduce_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def remove_stopwords(text: str, preserve_negation: bool = False) -> str:
    keep = set(NEGATORS) if preserve_negation else set()
    tokens = text.split()
    return " ".join(t for t in tokens if (t not in STOP_WORDS) or (t in keep))


def lemmatize_texts(texts: list[str]) -> list[str]:
    out = []
    for doc in nlp.pipe(texts, batch_size=200):
        out.append(" ".join(tok.lemma_ for tok in doc))
    return out


def emoji_to_text(text: str) -> str:
    return emoji.demojize(text, delimiters=(" ", " "))


def variant_config(**overrides) -> dict:
    cfg = copy.deepcopy(b0_config())
    for k, v in overrides.items():
        cfg["preprocessing"][k] = v
    return cfg


def run_stage_experiment(name, tag_experiment_id, texts, labels, partitions_df,
                          indices, config, protocol_run_id, member_id="carlos.cardona02"):
    df = pd.DataFrame({"index": indices, "text_proc": texts, "label": labels})
    merged = df.merge(partitions_df, on="index", how="inner")

    scores = []
    for fold_id in range(3):
        train_part = merged[merged["fold"] != fold_id]
        val_part = merged[merged["fold"] == fold_id]
        pipe = Pipeline([
            ("vectorizer", CountVectorizer(ngram_range=(1, 1))),
            ("clf", LogisticRegression()),
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
        mlflow.set_tag("lab_stage", "preprocessing")
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

    base_clean = sample_df["text"].apply(preprocess_b0)
    results = {}

    print("=== P_STOPWORDS ===")
    texts = base_clean.apply(lambda t: remove_stopwords(t, preserve_negation=False))
    cfg = variant_config(stopwords="remove", negators=[])
    results["P_STOPWORDS"] = run_stage_experiment(
        "P_STOPWORDS", "P_STOPWORDS", texts, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("=== P_STOPWORDS_NEGATION ===")
    texts = base_clean.apply(lambda t: remove_stopwords(t, preserve_negation=True))
    cfg = variant_config(stopwords="remove_preserve_negation", negators=NEGATORS)
    results["P_STOPWORDS_NEGATION"] = run_stage_experiment(
        "P_STOPWORDS_NEGATION", "P_STOPWORDS_NEGATION", texts, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("=== P_LEMMA ===")
    texts = pd.Series(lemmatize_texts(base_clean.tolist()), index=base_clean.index)
    cfg = variant_config(lemmatize=True)
    results["P_LEMMA"] = run_stage_experiment(
        "P_LEMMA", "P_LEMMA", texts, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("=== P_ELONGATION ===")
    texts = base_clean.apply(reduce_elongation)
    cfg = variant_config(elongation="normalize", elongation_spec="reduce_repeated_chars_to_2")
    results["P_ELONGATION"] = run_stage_experiment(
        "P_ELONGATION", "P_ELONGATION", texts, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("=== P_EMOJI ===")
    texts = base_clean.apply(emoji_to_text)
    cfg = variant_config(emoji="text", emoji_spec="emoji.demojize")
    results["P_EMOJI"] = run_stage_experiment(
        "P_EMOJI", "P_EMOJI", texts, sample_df["label"], partitions_df,
        sample_df["index"], cfg, protocol_run_id,
    )

    print("\n=== RESUMEN DE LA ETAPA DE PREPROCESAMIENTO ===")
    for name, (run_id, metrics) in results.items():
        print(f"{name:22s} macro_f1_mean={metrics['macro_f1_mean']:.4f}  run_id={run_id}")
    print("\nCompara estos valores con B0 y elige cuál preprocesamiento pasa a la etapa")
    print("de Representación (puedes quedarte con B0 si ninguno mejora).")


if __name__ == "__main__":
    main()
