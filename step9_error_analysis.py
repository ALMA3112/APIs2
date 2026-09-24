"""
Análisis de errores (Sección 5 / A.6). Selecciona aleatoriamente al menos 20
errores del modelo final (semilla 42), incluyendo ambas clases cuando existan,
los clasifica por categoría, y genera reports/error_analysis.csv y
reports/error_analysis.md.
"""
import csv
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import mlflow

from common import set_tracking, DATASET_ID, DATASET_REVISION

RANDOM_SEED = 42
TARGET_N = 20

POS_WORDS = {"love", "good", "great", "amazing", "happy", "best", "awesome", "nice",
             "wonderful", "fantastic", "excellent", "glad", "enjoy", "fun", "perfect"}
NEG_WORDS = {"hate", "bad", "terrible", "worst", "sad", "awful", "horrible", "annoying",
             "sucks", "boring", "disappointed", "angry", "upset", "poor"}
NEGATORS = {"no", "not", "n't", "never", "none", "nobody", "nothing", "nowhere", "neither",
            "nor", "cannot"}
CONTRAST = {"but", "however", "although", "though", "yet", "except"}
INTENSIFIERS = {"very", "so", "really", "extremely", "totally", "absolutely", "super",
                 "incredibly", "freaking", "damn"}
SARCASM_PHRASES = ["yeah right", "oh great", "just what i needed", "thanks a lot",
                    "great, just great", "oh joy", "wow, just wow", "can't wait", "/s"]
INFORMAL_TOKENS = {"u", "ur", "lol", "omg", "gonna", "wanna", "gotta", "kinda", "idk",
                    "lmao", "smh", "tbh", "btw", "plz", "pls"}
HASHTAG_RE = re.compile(r"#\w+")
ELONG_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"[a-z']+")

try:
    import emoji as emoji_lib

    def has_emoji(t: str) -> bool:
        return len(emoji_lib.emoji_list(t)) > 0
except Exception:
    def has_emoji(t: str) -> bool:
        return False


def categorize(text: str) -> str:
    t = text.lower()
    tokens = set(TOKEN_RE.findall(t))

    if has_emoji(text):
        return "emoji"
    if HASHTAG_RE.search(text):
        return "hashtag"
    if ELONG_RE.search(text):
        return "elongation"
    if any(p in t for p in SARCASM_PHRASES):
        return "sarcasm"
    if tokens & NEGATORS:
        return "negation"
    if tokens & CONTRAST:
        return "contrast"
    if tokens & INTENSIFIERS:
        return "intensification"
    if tokens & INFORMAL_TOKENS:
        return "informal"
    if (tokens & POS_WORDS) and (tokens & NEG_WORDS):
        return "mixed"
    return "other"


def select_sample(error_idx: np.ndarray, true_labels: pd.Series, rng: np.random.Generator) -> np.ndarray:
    neg_err = np.array([i for i in error_idx if true_labels.iloc[i] == "negative"])
    pos_err = np.array([i for i in error_idx if true_labels.iloc[i] == "positive"])

    if len(neg_err) > 0 and len(pos_err) > 0:
        half = TARGET_N // 2
        chosen_neg = rng.choice(neg_err, size=min(half, len(neg_err)), replace=False)
        chosen_pos = rng.choice(pos_err, size=min(TARGET_N - len(chosen_neg), len(pos_err)), replace=False)
        sample = np.concatenate([chosen_neg, chosen_pos])
        if len(sample) < TARGET_N:
            remaining = np.array([i for i in error_idx if i not in sample])
            if len(remaining) > 0:
                extra = rng.choice(remaining, size=min(TARGET_N - len(sample), len(remaining)), replace=False)
                sample = np.concatenate([sample, extra])
    else:
        sample = rng.choice(error_idx, size=min(TARGET_N, len(error_idx)), replace=False)

    return np.sort(np.unique(sample))


def main():
    set_tracking()
    print(f"Cargando conjunto test de {DATASET_ID}@{DATASET_REVISION} ...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    test_df = ds["test"].to_pandas().reset_index(drop=True)
    label_col = "label" if "label" in test_df.columns else "sentiment"
    label_map = {0: "negative", 1: "positive"}

    print("Cargando modelo sentiment140@champion ...")
    model = mlflow.pyfunc.load_model("models:/sentiment140@champion")

    print(f"Prediciendo sobre los {len(test_df)} registros de test (puede tardar) ...")
    preds = model.predict(pd.DataFrame({"text": test_df["text"].tolist()}))
    preds = list(preds)
    true_labels = test_df[label_col].map(label_map)

    errors_mask = np.array(preds) != true_labels.to_numpy()
    error_idx = np.where(errors_mask)[0]
    print(f"Total de errores encontrados: {len(error_idx)} de {len(test_df)} "
          f"({len(error_idx) / len(test_df) * 100:.2f}%)")

    rng = np.random.default_rng(RANDOM_SEED)
    sample = select_sample(error_idx, true_labels, rng)
    print(f"Casos seleccionados para el análisis: {len(sample)}")

    rows = []
    for i in sample:
        i = int(i)
        text = test_df.loc[i, "text"]
        rows.append({
            "index": i,
            "text": text,
            "true_label": true_labels.iloc[i],
            "predicted_label": preds[i],
            "category": categorize(text),
        })

    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/error_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "text", "true_label", "predicted_label", "category"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Escrito {csv_path} con {len(rows)} filas")

    freq = Counter(r["category"] for r in rows)
    freq_sorted = freq.most_common()

    md_lines = []
    md_lines.append("# Análisis de errores — sentiment140\n\n")
    md_lines.append(f"Total de errores en test: {len(error_idx)} de {len(test_df)} "
                     f"({len(error_idx) / len(test_df) * 100:.2f}%).\n\n")
    md_lines.append(f"Casos analizados (muestra aleatoria, semilla {RANDOM_SEED}): {len(rows)}.\n\n")
    md_lines.append("## Frecuencia por categoría\n\n")
    md_lines.append("| Categoría | Frecuencia |\n|---|---|\n")
    for cat, n in freq_sorted:
        md_lines.append(f"| {cat} | {n} |\n")

    md_lines.append("\n## Interpretación\n\n")
    if len(freq_sorted) == 1:
        cat, n = freq_sorted[0]
        md_lines.append(
            f"Todos los errores analizados pertenecen a la categoría **{cat}** "
            f"({n} de {len(rows)} casos).\n\n"
            f"[EDITA ESTO: explica por qué el modelo falla sistemáticamente en este "
            f"fenómeno, con 1-2 ejemplos concretos tomados del CSV.]\n"
        )
    else:
        (cat1, n1), (cat2, n2) = freq_sorted[0], freq_sorted[1]
        md_lines.append(
            f"Los dos patrones más frecuentes son **{cat1}** ({n1} casos) y "
            f"**{cat2}** ({n2} casos).\n\n"
            f"[EDITA ESTO: explica qué mecanismo del pipeline (TF-IDF unigramas+bigramas "
            f"con Logistic Regression) explica cada patrón -- por ejemplo, incapacidad de "
            f"capturar negación de largo alcance, sarcasmo sin señales léxicas claras, "
            f"mezcla de sentimientos en un mismo tweet, etc. Usa 1-2 ejemplos concretos "
            f"del CSV para cada patrón.]\n"
        )

    md_path = "reports/error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.writelines(md_lines)
    print(f"Escrito {md_path}")

    print("\nIMPORTANTE: abre reports/error_analysis.md y reemplaza el texto entre [EDITA")
    print("ESTO: ...] con tu interpretación real, basada en los casos del CSV, antes de")
    print("registrar estos archivos como artefactos del run final.")


if __name__ == "__main__":
    main()
