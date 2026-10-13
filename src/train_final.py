import os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from model_def import b0_prep

REV = "b6037e127257d95b9b23d31f78b264b9ebe697fd"
OUT = os.path.expanduser("~/SageMaker/lab2/artifacts"); os.makedirs(OUT, exist_ok=True)
REP = os.path.expanduser("~/SageMaker/lab2/reports"); os.makedirs(REP, exist_ok=True)
if os.path.exists(f"{OUT}/final_results.json"):
    raise SystemExit("final_results.json ya existe: test ya fue evaluado. No se repite.")
assert sklearn.__version__ == "1.5.2", sklearn.__version__

ds = load_dataset("adilbekovich/Sentiment140Twitter", revision=REV)
tr, te = ds["train"].to_pandas(), ds["test"].to_pandas()
assert len(tr) == 1_360_000 and len(te) == 240_000

print("vectorizando train...", flush=True)
vec = TfidfVectorizer(ngram_range=(1, 2))
Xtr = vec.fit_transform([b0_prep(t) for t in tr["text"]])
print("features:", Xtr.shape, flush=True)
clf = LogisticRegression().fit(Xtr, tr["label"])
joblib.dump({"vec": vec, "clf": clf}, f"{OUT}/final_pipeline.joblib", compress=3)

print("evaluando test (una sola vez)...", flush=True)
pred = clf.predict(vec.transform([b0_prep(t) for t in te["text"]]))
f1 = float(f1_score(te["label"], pred, average="macro"))
print("TEST macro_f1 =", f1, flush=True)

# ---- muestra aleatoria de 40 errores (semilla 42) ----
te["index"] = np.arange(len(te)); te["pred"] = pred
err = te[te["label"] != te["pred"]]
s = err.sample(n=40, random_state=42).sort_values("index")
assert set(s["label"]) == {0, 1}, "la muestra debe incluir errores de ambas clases"

def categoria(t):
    low = t.lower()
    if re.search(r"[:;=8][-o^']?[()dDpP/\\|]|[\U0001F300-\U0001FAFF]", t): return "emoji"
    if re.search(r"#\w+", t): return "hashtag"
    if re.search(r"\b(yeah right|as if|oh great|sure thing|thanks a lot)\b", low): return "sarcasm"
    if re.search(r"\b(no|not|never|nothing|nobody|none|cannot|dont|cant|wont|didnt|isnt|wasnt|doesnt|havent)\b|n't", low): return "negation"
    if re.search(r"\b(but|however|although|though|yet|except)\b", low): return "contrast"
    if re.search(r"\b(so|very|really|too|super|totally|extremely)\b|!{2,}|\b[A-Z]{3,}\b", t): return "intensification"
    if re.search(r"(\w)\1{2,}", low): return "elongation"
    if re.search(r"\b(u|ur|r|lol|omg|gonna|wanna|gotta|idk|smh|lmao|thx|pls|plz)\b", low): return "informal"
    return "other"

L = {0: "negative", 1: "positive"}
d = pd.DataFrame({"index": s["index"].values, "text": s["text"].values,
                  "true_label": s["label"].map(L).values, "predicted_label": s["pred"].map(L).values})
d["category"] = [categoria(t) for t in d["text"]]
d.to_csv(f"{REP}/error_analysis.csv", index=False, encoding="utf-8")

json.dump({"test_macro_f1": f1, "training_size": int(len(tr)), "test_size": int(len(te)),
           "errors_total": int(len(err)), "sklearn_version": sklearn.__version__,
           "n_features": int(Xtr.shape[1])}, open(f"{OUT}/final_results.json", "w"), indent=1)
print(d["category"].value_counts().to_string())
print("listo.")
