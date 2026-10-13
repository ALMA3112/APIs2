import os, io, csv, json, functools
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests, mlflow, mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from requests.exceptions import RequestException
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT, MODEL, ALIAS = "nlp-lab2-sentiment140", "sentiment140", "champion"
mlflow.set_tracking_uri(URI)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    try: load_champion()
    except Exception: pass
    yield

app = FastAPI(title="Sentiment140 API", lifespan=lifespan)
_cache = {}

STAGE = {"T0": "reference", "B0": "baseline", "ABLATION": "ablation"}
STAGE.update({k: "preprocessing" for k in ["P_STOPWORDS", "P_STOPWORDS_NEGATION", "P_LEMMA", "P_ELONGATION", "P_EMOJI"]})
STAGE.update({k: "representation" for k in ["R_BOW", "R_TFIDF_UNI", "R_TFIDF_UNI_BI", "R_SPACY"]})
STAGE.update({k: "classifier" for k in ["C_LOGREG", "C_LINEAR_SVM", "C_SGD"]})
EXTRA_STAGES = {"preprocessing", "representation", "classifier", "ablation"}
SIX = {"preprocessing.stopwords", "preprocessing.lemmatize", "preprocessing.elongation",
       "preprocessing.emoji", "representation", "classifier"}
PRE = {"lowercase","url","mention","whitespace","stopwords","negators","lemmatize","elongation",
       "elongation_spec","emoji","emoji_spec","resources","additional"}
REP = {"type","ngram_range","library","library_version","spacy_model","spacy_model_version","parameters"}
CLF = {"type","library","library_version","parameters"}
EXP_TAGS = ["lab_protocol_run_id","lab_experiment_id","lab_stage","lab_member_id","lab_configuration_id","notebook_arn"]
FIN_TAGS = ["lab_protocol_run_id","lab_selected_experiment_run_id","lab_configuration_id","lab_member_id","notebook_arn"]
METRICS = ["macro_f1_fold_0","macro_f1_fold_1","macro_f1_fold_2","macro_f1_mean","macro_f1_std"]
PROV = "provenance/sagemaker-resource-metadata.json"

# ---------------- utilidades MLflow ----------------
def audit(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        try:
            requests.get(URI.rstrip("/") + "/health", timeout=3).raise_for_status()
            return fn(*a, **k)
        except HTTPException:
            raise
        except (RequestException, MlflowException):
            raise HTTPException(status_code=503, detail="mlflow_unavailable")
    return w

def list_arts(c, rid, path=None):
    out = []
    for a in c.list_artifacts(rid, path):
        out += list_arts(c, rid, a.path) if a.is_dir else [a.path]
    return sorted(out)

def load_json(rid, path):
    return json.loads(mlflow.artifacts.load_text(f"runs:/{rid}/{path}"))

def presented(c):
    e = c.get_experiment_by_name(EXPERIMENT)
    if e is None or e.lifecycle_stage != "active":
        return []
    out, token = [], None
    while True:
        page = c.search_runs([e.experiment_id], max_results=1000, page_token=token)
        out += list(page)
        token = page.token
        if not token:
            break
    return [r for r in out if r.data.tags.get("lab_run_type") in ("protocol", "experiment", "final")]

def info(run):
    c = MlflowClient(tracking_uri=URI)
    rid = run.info.run_id
    arts = list_arts(c, rid)
    cfg = prov = None
    if run.data.tags["lab_run_type"] != "protocol" and "run/configuration.json" in arts:
        try: cfg = load_json(rid, "run/configuration.json")
        except Exception: cfg = None
    if PROV in arts:
        try: prov = load_json(rid, PROV)
        except Exception: prov = None
    return {"run": run, "id": rid, "tags": run.data.tags, "params": run.data.params,
            "metrics": run.data.metrics, "arts": arts, "cfg": cfg, "prov": prov}

# ---------------- reglas de validez ----------------
def cfg_ok(cfg, eid):
    if not isinstance(cfg, dict) or set(cfg) != {"preprocessing", "representation", "classifier"}:
        return False
    if eid == "T0":
        return cfg["preprocessing"] is None and cfg["representation"] is None \
            and isinstance(cfg["classifier"], dict) and cfg["classifier"].get("type") == "most_frequent"
    pr, rp, cl = cfg["preprocessing"], cfg["representation"], cfg["classifier"]
    if not (isinstance(pr, dict) and isinstance(rp, dict) and isinstance(cl, dict)): return False
    if set(pr) != PRE or set(rp) != REP or set(cl) != CLF: return False
    if (pr["stopwords"] == "remove_preserve_negation") != bool(pr["negators"]): return False
    if (pr["elongation"] == "normalize") != (pr["elongation_spec"] is not None): return False
    if (pr["emoji"] == "text") != (pr["emoji_spec"] is not None): return False
    if rp["type"] in ("bow", "tfidf"):
        if not (isinstance(rp["ngram_range"], list) and len(rp["ngram_range"]) == 2): return False
        if rp["spacy_model"] is not None or rp["spacy_model_version"] is not None: return False
    elif rp["type"] == "spacy_embedding":
        if rp["ngram_range"] != [] or not rp["parameters"].get("document_vector_method"): return False
    else:
        return False
    return cl["type"] in ("logistic_regression", "linear_svm", "sgd")

def attributable(d, members):
    t = d["tags"]
    arn = t.get("notebook_arn")
    return bool(arn) and members.get(t.get("lab_member_id")) == arn \
        and isinstance(d["prov"], dict) and d["prov"].get("ResourceArn") == arn

def exp_valid(d, pid, members, byid):
    r, t, m = d["run"], d["tags"], d["metrics"]
    if r.info.status != "FINISHED": return False
    if any(not t.get(k) for k in EXP_TAGS) or t["lab_protocol_run_id"] != pid: return False
    eid = t["lab_experiment_id"]
    if eid in STAGE:
        if t["lab_stage"] != STAGE[eid]: return False
    elif eid != "EXTRA" or t["lab_stage"] not in EXTRA_STAGES:
        return False
    if any(k not in m for k in METRICS): return False
    f3 = [m[f"macro_f1_fold_{i}"] for i in range(3)]
    if abs(np.mean(f3) - m["macro_f1_mean"]) > 1e-6 or abs(np.std(f3, ddof=0) - m["macro_f1_std"]) > 1e-6: return False
    if d["cfg"] is None or not cfg_ok(d["cfg"], eid): return False
    if not attributable(d, members): return False
    if t["lab_stage"] == "ablation" or eid == "ABLATION":
        par = byid.get(t.get("lab_ablation_parent_run_id"))
        if par is None or d["params"].get("ablation_reverted_decision") not in SIX or "macro_f1_delta" not in m: return False
        if abs((par["metrics"].get("macro_f1_mean", 1e9) - m["macro_f1_mean"]) - m["macro_f1_delta"]) > 1e-6: return False
    return True

def final_valid(d, pid, members, byid, valid):
    r, t, m = d["run"], d["tags"], d["metrics"]
    if r.info.status != "FINISHED": return False
    if any(not t.get(k) for k in FIN_TAGS) or t["lab_protocol_run_id"] != pid: return False
    if d["params"].get("training_size") != "1360000" or "test_macro_f1" not in m: return False
    need = {"run/configuration.json", PROV, "reports/error_analysis.csv", "reports/error_analysis.md"}
    if not need <= set(d["arts"]): return False
    if d["cfg"] is None or not cfg_ok(d["cfg"], "FINAL") or not attributable(d, members): return False
    sel = byid.get(t["lab_selected_experiment_run_id"])
    if sel is None or not valid.get(sel["id"]): return False
    return sel["tags"].get("lab_configuration_id") == t["lab_configuration_id"] and sel["cfg"] == d["cfg"]

def collect():
    c = MlflowClient(tracking_uri=URI)
    runs = presented(c)
    with ThreadPoolExecutor(8) as ex:
        infos = list(ex.map(info, runs))
    prots = [d for d in infos if d["tags"]["lab_run_type"] == "protocol"]
    pid = prots[0]["id"] if len(prots) == 1 else None
    members = {}
    if pid and "protocol/members.csv" in prots[0]["arts"]:
        text = mlflow.artifacts.load_text(f"runs:/{pid}/protocol/members.csv")
        members = {row["member_id"]: row["notebook_arn"] for row in csv.DictReader(io.StringIO(text))}
    byid = {d["id"]: d for d in infos}
    valid = {}
    for d in infos:
        if d["tags"]["lab_run_type"] == "experiment":
            valid[d["id"]] = exp_valid(d, pid, members, byid)
    for d in infos:
        if d["tags"]["lab_run_type"] == "final":
            valid[d["id"]] = final_valid(d, pid, members, byid, valid)
    return {"infos": infos, "prots": prots, "pid": pid, "members": members, "valid": valid, "byid": byid}

# ---------------- modelo ----------------
def champion_version():
    c = MlflowClient(tracking_uri=URI)
    try:
        return c.get_model_version_by_alias(MODEL, ALIAS)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise

def load_champion():
    try:
        mv = champion_version()
        if mv is None: raise RuntimeError("sin alias")
        v = mv.version
        if v not in _cache:
            _cache[v] = (mv.run_id, mlflow.pyfunc.load_model(f"models:/{MODEL}/{v}"))
        return _cache[v]
    except Exception:
        raise HTTPException(status_code=503, detail="model_unavailable")

# ---------------- endpoints ----------------
@app.post("/api/v1/predict")
async def predict(request: Request):
    try: body = await request.json()
    except Exception: raise HTTPException(status_code=422, detail="invalid_json")
    if not isinstance(body, dict) or "text" not in body:
        raise HTTPException(status_code=422, detail="invalid_input")
    t = body["text"]
    texts = [t] if isinstance(t, str) else t
    if not isinstance(texts, list) or not 1 <= len(texts) <= 32:
        raise HTTPException(status_code=422, detail="invalid_input")
    for x in texts:
        if not isinstance(x, str) or not x.strip() or len(x) > 1000:
            raise HTTPException(status_code=422, detail="invalid_input")
    rid, model = load_champion()
    return {"model_run_id": rid, "predictions": [str(p) for p in model.predict(texts)]}

@app.get("/health")
def health():
    try:
        rid, model = load_champion()
        model.predict(["health check"])
        return {"status": "ok", "model_run_id": rid}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_run_id": None})

@app.get("/audit/protocol")
@audit
def audit_protocol():
    c = MlflowClient(tracking_uri=URI)
    prots = [r for r in presented(c) if r.data.tags["lab_run_type"] == "protocol"]
    if len(prots) != 1:
        raise HTTPException(status_code=409, detail="protocol_not_unique")
    r = prots[0]; p = r.data.params
    return {"protocol_run_id": r.info.run_id, "dataset_id": p.get("dataset_id"),
            "dataset_revision": p.get("dataset_revision"), "sampling_strategy": p.get("sampling_strategy"),
            "sample_size": int(p["sample_size"]), "random_seed": int(p["random_seed"]),
            "cv_strategy": p.get("cv_strategy"), "cv_folds": int(p["cv_folds"]),
            "cv_shuffle": p.get("cv_shuffle", "").lower() == "true",
            "partitions_artifact": "protocol/partitions.csv", "members_artifact": "protocol/members.csv"}

@app.get("/audit/runs")
@audit
def audit_runs():
    S = collect()
    out = []
    for d in sorted(S["infos"], key=lambda d: d["id"]):
        out.append({"run_id": d["id"], "status": d["run"].info.status, "run_type": d["tags"]["lab_run_type"],
                    "params": dict(d["params"]), "metrics": {k: float(v) for k, v in d["metrics"].items()},
                    "tags": dict(d["tags"]), "artifacts": d["arts"], "configuration": d["cfg"]})
    return {"runs": out}

@app.get("/audit/contributions")
@audit
def audit_contributions():
    S = collect(); infos, valid, members = S["infos"], S["valid"], S["members"]
    exps = [d for d in infos if d["tags"]["lab_run_type"] == "experiment"]
    finals = [d for d in infos if d["tags"]["lab_run_type"] == "final"]
    out = []
    for mid in sorted(members):
        mine = [d for d in exps if valid[d["id"]] and d["tags"].get("lab_member_id") == mid
                and d["tags"].get("notebook_arn") == members[mid]]
        counted = [d for d in mine if d["tags"]["lab_experiment_id"] not in ("T0", "B0")]
        cids = sorted({d["tags"]["lab_configuration_id"] for d in counted})
        out.append({"member_id": mid, "notebook_arn": members[mid],
                    "run_ids": sorted(d["id"] for d in mine),
                    "counted_run_ids": sorted(d["id"] for d in counted),
                    "configuration_ids": cids, "valid_configurations": len(cids),
                    "stages": sorted({d["tags"]["lab_stage"] for d in counted})})
    invalid = sorted(d["id"] for d in exps + finals if not valid[d["id"]])
    unattr = sorted(d["id"] for d in exps + finals if not attributable(d, members))
    return {"members": out, "invalid_run_ids": invalid, "unattributed_run_ids": unattr}

@app.get("/audit/model")
@audit
def audit_model():
    mv = champion_version()
    if mv is None:
        raise HTTPException(status_code=404, detail="champion_not_found")
    S = collect(); d = S["byid"].get(mv.run_id)
    if d is None or d["tags"]["lab_run_type"] != "final" or not S["valid"].get(d["id"]):
        raise HTTPException(status_code=409, detail="champion_invalid")
    t = d["tags"]
    return {"model_name": MODEL, "alias": ALIAS, "version": int(mv.version), "run_id": d["id"],
            "protocol_run_id": t["lab_protocol_run_id"], "selected_experiment_run_id": t["lab_selected_experiment_run_id"],
            "configuration_id": t["lab_configuration_id"], "configuration": d["cfg"],
            "training_size": int(d["params"]["training_size"]), "test_macro_f1": float(d["metrics"]["test_macro_f1"])}
