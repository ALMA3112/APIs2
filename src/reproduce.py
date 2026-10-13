import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import *

def make(kind):
    if kind == "B0":   return CountVectorizer(), LogisticRegression()
    if kind == "CAND": return TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression()

def cv(sample, kind):
    X = [b0_prep(t) for t in sample["text"]]
    X = np.array(X, dtype=object); y = sample["label"].values; f = sample["fold"].values
    out = []
    for k in range(3):
        tr, va = f != k, f == k
        vec, clf = make(kind)
        clf.fit(vec.fit_transform(X[tr]), y[tr])
        out.append(float(f1_score(y[va], clf.predict(vec.transform(X[va])), average="macro")))
        print(kind, "fold", k, out[-1], flush=True)
    return out

if __name__ == "__main__":
    c = client()
    exp = c.get_experiment_by_name(EXPERIMENT)
    reg = {r.data.tags["lab_experiment_id"]: r.data.metrics for r in
           c.search_runs([exp.experiment_id], filter_string="tags.lab_run_type = 'experiment'", max_results=500)}
    sample = sample_with_folds(load_train(), load_partitions(c))
    print("muestra:", len(sample), sample["label"].value_counts().to_dict())
    for kind, exp_id in [("B0", "B0"), ("CAND", "R_TFIDF_UNI_BI")]:
        mine = cv(sample, kind); m = reg[exp_id]
        ref = [m[f"macro_f1_fold_{i}"] for i in range(3)]
        d = max(abs(a - b) for a, b in zip(mine, ref))
        print(f"== {kind}: mio={np.round(mine,5)} registrado={np.round(ref,5)} max|dif|={d:.2e} ->", "REPRODUCE" if d < 1e-4 else "NO COINCIDE")
