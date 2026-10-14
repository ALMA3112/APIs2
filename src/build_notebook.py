import json, os
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("# Auditoría del experimento: Sentiment140\nReconstruye desde el MLflow Tracking Server (sin copias locales) el protocolo, los runs, las ablaciones, las contribuciones y la cadena `sentiment140@champion` → run final → run experimental seleccionado.")
code('''import os, json, numpy as np, pandas as pd, mlflow
from IPython.display import display
from mlflow.tracking import MlflowClient
URI = os.environ.get("MLFLOW_TRACKING_URI", "http://34.228.215.204:5000")
mlflow.set_tracking_uri(URI); c = MlflowClient()
exp = c.get_experiment_by_name("nlp-lab2-sentiment140")
runs = [r for r in c.search_runs([exp.experiment_id], max_results=1000) if r.data.tags.get("lab_run_type")]
print(URI, "|", exp.name, exp.lifecycle_stage, "| mlflow", mlflow.__version__)
print(pd.Series([r.data.tags["lab_run_type"] for r in runs]).value_counts().to_dict())
def dl(rid, p): return mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=p)''')
md("## 1. Protocolo")
code('''prot = [r for r in runs if r.data.tags["lab_run_type"] == "protocol"]
assert len(prot) == 1, "debe existir exactamente un protocolo"
pid = prot[0].info.run_id
print("protocol_run_id:", pid); print(json.dumps(prot[0].data.params, indent=1))
parts = pd.read_csv(dl(pid, "protocol/partitions.csv"))
print("partitions:", len(parts), "filas | índices únicos:", parts["index"].is_unique,
      "| ascendente:", parts["index"].is_monotonic_increasing, "| folds:", parts["fold"].value_counts().sort_index().to_dict())
display(pd.read_csv(dl(pid, "protocol/members.csv")))''')
md("## 2. Runs experimentales y consistencia aritmética de las métricas")
code('''rows = []
for r in runs:
    t, m = r.data.tags, r.data.metrics
    if t["lab_run_type"] != "experiment": continue
    f = [m[f"macro_f1_fold_{i}"] for i in range(3)]
    rows.append(dict(experimento=t["lab_experiment_id"], etapa=t["lab_stage"], miembro=t["lab_member_id"],
                     config=t["lab_configuration_id"], run=r.info.run_id[:8], fold0=f[0], fold1=f[1], fold2=f[2],
                     media=m["macro_f1_mean"], std=m["macro_f1_std"],
                     media_ok=abs(np.mean(f) - m["macro_f1_mean"]) < 1e-6,
                     std_poblacional_ok=abs(np.std(f, ddof=0) - m["macro_f1_std"]) < 1e-6,
                     protocolo_ok=t["lab_protocol_run_id"] == pid))
df = pd.DataFrame(rows).sort_values(["etapa", "media"], ascending=[True, False])
display(df.round(5)); print("todas las métricas consistentes:", bool(df[["media_ok", "std_poblacional_ok", "protocolo_ok"]].all().all()))''')
md("## 3. Qué cambia cada configuración frente a B0")
code('''def flat(d, p=""):
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict) and v: o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
cfgs = {r.data.tags["lab_experiment_id"]: json.load(open(dl(r.info.run_id, "run/configuration.json")))
        for r in runs if r.data.tags["lab_run_type"] == "experiment"}
fb = flat(cfgs["B0"])
for k, cfg in cfgs.items():
    if k in ("T0", "B0"): continue
    fc = flat(cfg)
    print(f"{k:22}", sorted(x for x in set(fb) | set(fc) if fb.get(x) != fc.get(x) and not x.startswith("preprocessing.resources")))''')
md("## 4. Ablación")
code('''by = {r.data.tags.get("lab_experiment_id"): r for r in runs if r.data.tags["lab_run_type"] == "experiment"}
a = by["ABLATION"]; par = c.get_run(a.data.tags["lab_ablation_parent_run_id"])
print("decisión revertida:", a.data.params["ablation_reverted_decision"])
print("padre (candidato):", par.info.run_id, par.data.tags["lab_experiment_id"], "| config", par.data.tags["lab_configuration_id"])
d = par.data.metrics["macro_f1_mean"] - a.data.metrics["macro_f1_mean"]
print(f"media padre {par.data.metrics['macro_f1_mean']:.5f} - media ablación {a.data.metrics['macro_f1_mean']:.5f} = {d:.6f}",
      "| registrado:", round(a.data.metrics["macro_f1_delta"], 6), "| coincide:", abs(d - a.data.metrics["macro_f1_delta"]) < 1e-6)''')
md("## 5. Contribuciones por integrante")
code('''contrib = df[~df["experimento"].isin(["T0", "B0"])].groupby("miembro").agg(
    runs=("run", "count"), configuraciones=("config", "nunique"), etapas=("etapa", lambda s: sorted(set(s))))
contrib["cumple_minimo"] = (contrib["configuraciones"] >= 3) & (contrib["etapas"].map(len) >= 2)
display(contrib)''')
md("## 6. Cadena de trazabilidad: alias → run final → run experimental seleccionado")
code('''mv = c.get_model_version_by_alias("sentiment140", "champion")
fin = c.get_run(mv.run_id); ft = fin.data.tags
sel = c.get_run(ft["lab_selected_experiment_run_id"])
cf = json.load(open(dl(fin.info.run_id, "run/configuration.json"))); cs = json.load(open(dl(sel.info.run_id, "run/configuration.json")))
print("sentiment140@champion -> versión", mv.version, "-> run final", fin.info.run_id, "-> experimental", sel.info.run_id, f"({sel.data.tags['lab_experiment_id']})")
print("tipo de run:", ft["lab_run_type"], "| estado:", fin.info.status, "| training_size:", fin.data.params["training_size"], "| test_macro_f1:", fin.data.metrics["test_macro_f1"])
print("mismo configuration_id:", ft["lab_configuration_id"] == sel.data.tags["lab_configuration_id"], "| configuraciones equivalentes:", cf == cs)
print("run de protocolo referenciado es el único:", ft["lab_protocol_run_id"] == pid)
print("artefactos del run final:", sorted(x.path for x in c.list_artifacts(fin.info.run_id) if x.path != "model") + ["+ model/"])
prov = json.load(open(dl(fin.info.run_id, "provenance/sagemaker-resource-metadata.json")))
print("ResourceArn == notebook_arn:", prov["ResourceArn"] == ft["notebook_arn"], "|", ft["notebook_arn"])''')
md("## 7. API desplegada (opcional)\nCompara lo que responde la API con lo que muestra MLflow.")
code('''import requests
API = os.environ.get("API_URL", "http://34.228.215.204:8000")
try:
    h = requests.get(API + "/health", timeout=10).json(); m = requests.get(API + "/audit/model", timeout=60).json()
    p = requests.post(API + "/api/v1/predict", json={"text": ["i loved it", "worst day ever"]}, timeout=30).json()
    print(h, "\\n", p, "\\n audit/model run_id:", m["run_id"], "| coincide con el alias:", m["run_id"] == mv.run_id == h["model_run_id"] == p["model_run_id"])
except Exception as e:
    print("API no disponible:", e)''')

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "lab2", "display_name": "lab2", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.expanduser("~/SageMaker/lab2/notebooks/experiment_audit.ipynb")
json.dump(nb, open(out, "w"), indent=1); print("creado", out)
