import json, numpy as np, pandas as pd, mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://34.230.57.243:5000")
c = MlflowClient()
exp = c.get_experiment_by_name("nlp-lab2-sentiment140")
runs = c.search_runs([exp.experiment_id], max_results=1000)
sub = [r for r in runs if r.data.tags.get("lab_run_type") in ("protocol", "experiment", "final")]
P = []
def bad(r, m): P.append((r[:8] if r else "-", m))
def arts(rid, path=None):
    out = []
    for a in c.list_artifacts(rid, path):
        out += arts(rid, a.path) if a.is_dir else [a.path]
    return sorted(out)
def dl(rid, p): return mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=p)
def flat(d, p=""):
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict) and v: o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o

# ---------- protocolo ----------
prot = [r for r in sub if r.data.tags["lab_run_type"] == "protocol"]
if len(prot) != 1: bad(None, f"protocolos presentados = {len(prot)} (debe ser 1)")
pid = prot[0].info.run_id if prot else None
if prot:
    r = prot[0]; p = r.data.params
    exp_p = {"dataset_id": "adilbekovich/Sentiment140Twitter",
             "dataset_revision": "b6037e127257d95b9b23d31f78b264b9ebe697fd",
             "sampling_strategy": "stratified", "sample_size": "200000", "random_seed": "42",
             "cv_strategy": "StratifiedKFold", "cv_folds": "3", "cv_shuffle": "true"}
    for k, v in exp_p.items():
        if p.get(k) != v: bad(pid, f"param {k}={p.get(k)!r} (esperado {v!r})")
    if r.info.status != "FINISHED": bad(pid, "protocolo no FINISHED")
    a = arts(pid)
    for need in ("protocol/partitions.csv", "protocol/members.csv"):
        if need not in a: bad(pid, f"falta artefacto {need}")
    if "protocol/partitions.csv" in a:
        f = dl(pid, "protocol/partitions.csv")
        if open(f, encoding="utf-8").readline().strip() != "index,fold": bad(pid, "encabezado partitions")
        d = pd.read_csv(f)
        if len(d) != 200000: bad(pid, f"partitions filas={len(d)}")
        if not d["index"].is_unique or not d["index"].is_monotonic_increasing: bad(pid, "index no único/ascendente")
        if set(d["fold"].unique()) != {0, 1, 2}: bad(pid, "folds distintos de {0,1,2}")
        print("partitions OK:", len(d), "filas; folds:", d["fold"].value_counts().sort_index().to_dict())
    if "protocol/members.csv" in a:
        f = dl(pid, "protocol/members.csv")
        print("members.csv:\n", open(f, encoding="utf-8").read())
        if open(f, encoding="utf-8").readline().strip() != "member_id,notebook_arn": bad(pid, "encabezado members")

# ---------- experimentales ----------
STAGE = {"T0": "reference", "B0": "baseline", "ABLATION": "ablation"}
for k in ["P_STOPWORDS", "P_STOPWORDS_NEGATION", "P_LEMMA", "P_ELONGATION", "P_EMOJI"]: STAGE[k] = "preprocessing"
for k in ["R_BOW", "R_TFIDF_UNI", "R_TFIDF_UNI_BI", "R_SPACY"]: STAGE[k] = "representation"
for k in ["C_LOGREG", "C_LINEAR_SVM", "C_SGD"]: STAGE[k] = "classifier"
ALLOWED = {"P_STOPWORDS": {"preprocessing.stopwords"},
           "P_STOPWORDS_NEGATION": {"preprocessing.stopwords", "preprocessing.negators"},
           "P_LEMMA": {"preprocessing.lemmatize"},
           "P_ELONGATION": {"preprocessing.elongation", "preprocessing.elongation_spec"},
           "P_EMOJI": {"preprocessing.emoji", "preprocessing.emoji_spec"}}
PRE = {"lowercase","url","mention","whitespace","stopwords","negators","lemmatize","elongation","elongation_spec","emoji","emoji_spec","resources","additional"}
REP = {"type","ngram_range","library","library_version","spacy_model","spacy_model_version","parameters"}
CLF = {"type","library","library_version","parameters"}
SIX = {"preprocessing.stopwords","preprocessing.lemmatize","preprocessing.elongation","preprocessing.emoji","representation","classifier"}
REQ_TAGS = ["lab_run_type","lab_protocol_run_id","lab_experiment_id","lab_stage","lab_member_id","lab_configuration_id","notebook_arn"]
REQ_M = ["macro_f1_fold_0","macro_f1_fold_1","macro_f1_fold_2","macro_f1_mean","macro_f1_std"]

def check_cfg(rid, cfg, exp_id):
    if set(cfg) != {"preprocessing", "representation", "classifier"}: bad(rid, "claves de primer nivel de configuration.json"); return
    if exp_id == "T0":
        if cfg["preprocessing"] is not None or cfg["representation"] is not None or cfg["classifier"].get("type") != "most_frequent":
            bad(rid, "T0 no es la forma canónica")
        return
    pr, rp, cl = cfg["preprocessing"], cfg["representation"], cfg["classifier"]
    if set(pr) != PRE: bad(rid, f"preprocessing claves: {set(pr) ^ PRE}")
    if set(rp) != REP: bad(rid, f"representation claves: {set(rp) ^ REP}")
    if set(cl) != CLF: bad(rid, f"classifier claves: {set(cl) ^ CLF}")
    if set(pr) != PRE or set(rp) != REP: return
    if (pr["stopwords"] == "remove_preserve_negation") != bool(pr["negators"]): bad(rid, "negators vs stopwords")
    if (pr["elongation"] == "normalize") != (pr["elongation_spec"] is not None): bad(rid, "elongation_spec vs elongation")
    if (pr["emoji"] == "text") != (pr["emoji_spec"] is not None): bad(rid, "emoji_spec vs emoji")
    if rp["type"] in ("bow", "tfidf"):
        if not (isinstance(rp["ngram_range"], list) and len(rp["ngram_range"]) == 2): bad(rid, "ngram_range")
        if rp["spacy_model"] is not None or rp["spacy_model_version"] is not None: bad(rid, "campos spaCy deben ser null")
    elif rp["type"] == "spacy_embedding":
        if rp["ngram_range"] != []: bad(rid, "ngram_range debe ser []")
        if not rp["parameters"].get("document_vector_method"): bad(rid, "falta document_vector_method")
    else: bad(rid, f"representation.type={rp['type']}")
    if cl["type"] not in ("logistic_regression", "linear_svm", "sgd"): bad(rid, f"classifier.type={cl['type']}")

exps = [r for r in sub if r.data.tags["lab_run_type"] == "experiment"]
cfgs = {}; byexp = {}
for r in exps:
    rid, t, m = r.info.run_id, r.data.tags, r.data.metrics
    eid = t.get("lab_experiment_id"); byexp.setdefault(eid, []).append(r)
    if r.info.status != "FINISHED": bad(rid, f"{eid}: status {r.info.status}")
    for k in REQ_TAGS:
        if not t.get(k): bad(rid, f"{eid}: falta tag {k}")
    if t.get("lab_protocol_run_id") != pid: bad(rid, f"{eid}: lab_protocol_run_id no apunta al protocolo")
    if eid in STAGE and t.get("lab_stage") != STAGE[eid]: bad(rid, f"{eid}: lab_stage={t.get('lab_stage')}")
    if eid == "EXTRA" and t.get("lab_stage") not in ("preprocessing","representation","classifier","ablation"): bad(rid, "EXTRA con stage inválido")
    if eid not in STAGE and eid != "EXTRA": bad(rid, f"lab_experiment_id desconocido: {eid}")
    for k in REQ_M:
        if k not in m: bad(rid, f"{eid}: falta métrica {k}")
    if all(k in m for k in REQ_M):
        f3 = [m["macro_f1_fold_%d" % i] for i in range(3)]
        if abs(np.mean(f3) - m["macro_f1_mean"]) > 1e-6: bad(rid, f"{eid}: mean no coincide ({np.mean(f3)} vs {m['macro_f1_mean']})")
        if abs(np.std(f3, ddof=0) - m["macro_f1_std"]) > 1e-6: bad(rid, f"{eid}: std no es poblacional ddof=0 ({np.std(f3, ddof=0)} vs {m['macro_f1_std']})")
    a = arts(rid)
    for need in ("run/configuration.json", "provenance/sagemaker-resource-metadata.json"):
        if need not in a: bad(rid, f"{eid}: falta {need}")
    if "provenance/sagemaker-resource-metadata.json" in a:
        arn = json.load(open(dl(rid, "provenance/sagemaker-resource-metadata.json"))).get("ResourceArn")
        if arn != t.get("notebook_arn"): bad(rid, f"{eid}: ResourceArn != tag notebook_arn")
    if "run/configuration.json" in a:
        try: cfg = json.load(open(dl(rid, "run/configuration.json")))
        except Exception as e: bad(rid, f"{eid}: configuration.json inválido"); continue
        cfgs[rid] = (eid, cfg); check_cfg(rid, cfg, eid)
    if eid == "ABLATION":
        par = t.get("lab_ablation_parent_run_id"); dec = r.data.params.get("ablation_reverted_decision")
        if not par: bad(rid, "ABLATION sin lab_ablation_parent_run_id")
        if dec not in SIX: bad(rid, f"ablation_reverted_decision={dec!r}")
        if "macro_f1_delta" not in m: bad(rid, "ABLATION sin macro_f1_delta")
        else:
            pm = c.get_run(par).data.metrics["macro_f1_mean"]
            if abs((pm - m["macro_f1_mean"]) - m["macro_f1_delta"]) > 1e-6: bad(rid, "macro_f1_delta != mean(parent) - mean(ablation)")

# comparaciones obligatorias presentes
for k in STAGE:
    if k not in byexp: bad(None, f"falta la comparación {k}")
    elif len(byexp[k]) > 1 and k != "ABLATION": bad(None, f"{k} duplicado ({len(byexp[k])} runs)")

# diferencias frente a B0 (A.4)
b0 = next((cfg for (e, cfg) in cfgs.values() if e == "B0"), None)
if b0:
    fb = flat(b0)
    for rid, (eid, cfg) in cfgs.items():
        if eid == "T0": continue
        fc = flat(cfg)
        diff = sorted(k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k) and not k.startswith("preprocessing.resources"))
        print(f"{eid:22} {rid[:8]} cfg={[r for r in exps if r.info.run_id==rid][0].data.tags.get('lab_configuration_id')} difiere de B0 en: {diff}")
        if eid in ALLOWED and not set(diff) <= ALLOWED[eid] | {"representation.library_version", "classifier.library_version"}:
            bad(rid, f"{eid}: difiere de B0 en campos no permitidos: {diff}")

# equivalencia: mismo configuration_id <=> misma configuración
seen = {}
for r in exps:
    rid = r.info.run_id
    if rid in cfgs:
        key = json.dumps({k: v for k, v in cfgs[rid][1].items()}, sort_keys=True) if cfgs[rid][0] != "T0" else "T0"
        seen.setdefault(key, set()).add(r.data.tags.get("lab_configuration_id"))
for key, ids in seen.items():
    if len(ids) > 1: bad(None, f"configuraciones idénticas con configuration_id distintos: {ids}")
ids2cfg = {}
for r in exps:
    if r.info.run_id in cfgs:
        ids2cfg.setdefault(r.data.tags.get("lab_configuration_id"), set()).add(json.dumps(cfgs[r.info.run_id][1], sort_keys=True))
for i, s in ids2cfg.items():
    if len(s) > 1: bad(None, f"configuration_id {i} usado con configuraciones distintas")

# contribución individual
for mem in sorted({r.data.tags.get("lab_member_id") for r in exps}):
    mine = [r for r in exps if r.data.tags.get("lab_member_id") == mem and r.data.tags.get("lab_experiment_id") not in ("T0", "B0") and r.info.run_id in cfgs]
    cid = sorted({r.data.tags["lab_configuration_id"] for r in mine}); st = sorted({r.data.tags["lab_stage"] for r in mine})
    print(f"\ncontribución {mem}: configuraciones válidas={len(cid)} {cid} etapas={st}")
    if len(cid) < 3 or len(st) < 2: bad(None, f"{mem} no cumple el mínimo (>=3 configuraciones y >=2 etapas)")

print("\n===== PROBLEMAS =====")
for rid, m in P: print(rid, "|", m)
print("TOTAL:", len(P), "->", "CUMPLE" if not P else "REVISAR")
