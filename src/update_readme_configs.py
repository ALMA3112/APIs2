import os, json, mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"]); c = MlflowClient()
exp = c.get_experiment_by_name("nlp-lab2-sentiment140")
runs = [r for r in c.search_runs([exp.experiment_id], max_results=1000) if r.data.tags.get("lab_run_type") == "experiment"]
by = {r.data.tags["lab_experiment_id"]: r for r in runs}
def cfg(r): return json.load(open(mlflow.artifacts.download_artifacts(run_id=r.info.run_id, artifact_path="run/configuration.json")))
def flat(d, p=""):
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict): o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
b0 = flat(cfg(by["B0"]))
order = ["T0","B0","P_STOPWORDS","P_STOPWORDS_NEGATION","P_LEMMA","P_ELONGATION","P_EMOJI",
         "R_BOW","R_TFIDF_UNI","R_TFIDF_UNI_BI","R_SPACY","C_LOGREG","C_LINEAR_SVM","C_SGD","ABLATION"]
rows = []
for k in order:
    r = by[k]; cid = r.data.tags["lab_configuration_id"]
    if k == "T0": diffs = "clasificador `most_frequent`, sin preprocesamiento ni representación"
    elif k == "B0": diffs = "configuración base (ver [Pipeline final](#pipeline-final))"
    else:
        f = flat(cfg(r)); keys = sorted(x for x in set(b0) | set(f) if b0.get(x) != f.get(x))
        diffs = "; ".join("`" + x + "` = `" + json.dumps(f.get(x), ensure_ascii=False) + "`" for x in keys) \
                or "sin diferencias: misma configuración que B0"
    d = diffs.replace("|", "\\|")
    rows.append(f"| `{k}` | `{cid}` | {d} |")
block = ("<!-- CONFIG-TABLE:START -->\n### Detalle de las configuraciones\n\n"
         "Valores extraídos de `run/configuration.json` de cada run; se muestran solo los campos que difieren de B0 "
         "(esquema del Anexo A.3).\n\n| Experimento | `configuration_id` | Diferencias frente a B0 |\n|---|---|---|\n"
         + "\n".join(rows) + "\n<!-- CONFIG-TABLE:END -->\n")
p = os.path.expanduser("~/SageMaker/lab2/README.md"); s = open(p, encoding="utf-8").read()
if "<!-- CONFIG-TABLE:START -->" in s:
    a = s.index("<!-- CONFIG-TABLE:START -->"); b = s.index("<!-- CONFIG-TABLE:END -->") + len("<!-- CONFIG-TABLE:END -->\n")
    s = s[:a] + block + s[b:]
else:
    anchor = "### Lectura de los resultados"; assert anchor in s
    s = s.replace(anchor, block + "\n" + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print(block)
