import sys, os, time, requests
from mlflow.tracking import MlflowClient
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
bad = []
def chk(name, ok, extra=""):
    print(("OK   " if ok else "FALLA"), name, extra)
    if not ok: bad.append(name)
def post(j): return requests.post(B + "/api/v1/predict", json=j, timeout=30)

t0 = time.time(); r = post({"text": "this movie was surprisingly good"}); dt = time.time() - t0
chk("predict string", r.status_code == 200 and len(r.json()["predictions"]) == 1, f"{r.text[:120]} ({dt:.1f}s)")
rid = r.json().get("model_run_id") if r.ok else None
r = post({"text": ["i loved it", "worst day ever", "not bad at all"]})
chk("predict lote conserva cantidad", r.status_code == 200 and len(r.json()["predictions"]) == 3, r.text[:120])
r = post({"text": ["good day"] * 32}); chk("predict 32 textos", r.status_code == 200 and len(r.json()["predictions"]) == 32)
r = post({"text": "a" * 1000}); chk("predict 1000 caracteres", r.status_code == 200)
casos = [("vacío {}", {}), ("null", {"text": None}), ("string vacío", {"text": ""}), ("solo espacios", {"text": "  "}),
         ("lista vacía", {"text": []}), ("elemento vacío", {"text": ["ok", ""]}), ("elemento no string", {"text": ["ok", 7]}),
         ("número", {"text": 7}), ("33 textos", {"text": ["x"] * 33}), ("1001 caracteres", {"text": "a" * 1001})]
for name, body in casos:
    r = post(body); chk(f"inválida {name} -> 4xx", 400 <= r.status_code < 500, str(r.status_code))
r = requests.post(B + "/api/v1/predict", data="no json", headers={"Content-Type": "application/json"})
chk("inválida JSON roto -> 4xx", 400 <= r.status_code < 500, str(r.status_code))

h = requests.get(B + "/health"); chk("health", h.status_code == 200 and h.json().get("status") == "ok", h.text)
for ep in ["/audit/protocol", "/audit/runs", "/audit/contributions", "/audit/model"]:
    t0 = time.time(); r = requests.get(B + ep, timeout=60)
    chk(f"{ep} 200", r.status_code == 200, f"{time.time()-t0:.1f}s " + ("" if r.ok else r.text[:150]))
m = requests.get(B + "/audit/model").json()
c = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
alias_run = c.get_model_version_by_alias("sentiment140", "champion").run_id
chk("mismo run final en predict/health/audit-model/alias",
    len({rid, h.json().get("model_run_id"), m.get("run_id"), alias_run}) == 1, f"{rid} {m.get('run_id')} {alias_run}")
runs = requests.get(B + "/audit/runs").json()["runs"]
fin = [x for x in runs if x["run_type"] == "final"]
print("artefactos del run final:", [a for a in fin[0]["artifacts"] if not a.startswith("model/")] if fin else None)
cs = requests.get(B + "/audit/contributions").json()
print("contribuciones:", {k: v for k, v in cs["members"][0].items() if k in ("member_id", "valid_configurations", "stages")},
      "| inválidos:", cs["invalid_run_ids"], "| sin atribuir:", cs["unattributed_run_ids"])
print("\nRESULTADO:", "TODO OK" if not bad else f"{len(bad)} fallas: {bad}")
