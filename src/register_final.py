import json, os, shutil, sys, tempfile
import mlflow, mlflow.pyfunc, sklearn, pandas, numpy, joblib
sys.path.insert(0, os.path.expanduser("~/SageMaker/lab2/src"))
from common import client, EXPERIMENT, protocol_run_id
import model_def

H = os.path.expanduser("~/SageMaker/lab2")
c = client()
exp = c.get_experiment_by_name(EXPERIMENT)
assert exp.lifecycle_stage == "active"
runs = c.search_runs([exp.experiment_id], max_results=1000)
assert not [r for r in runs if r.data.tags.get("lab_run_type") == "final"], "ya existe un run final"
sel = [r for r in runs if r.data.tags.get("lab_experiment_id") == "C_LOGREG"]
assert len(sel) == 1, f"C_LOGREG: {len(sel)}"
sel = sel[0]; t = sel.data.tags
pid = protocol_run_id(c)
res = json.load(open(f"{H}/artifacts/final_results.json"))
meta = "/opt/ml/metadata/resource-metadata.json"
arn = json.load(open(meta))["ResourceArn"]
assert arn == t["notebook_arn"], (arn, t["notebook_arn"])

tmp = tempfile.mkdtemp()
cfg = mlflow.artifacts.download_artifacts(run_id=sel.info.run_id, artifact_path="run/configuration.json", dst_path=tmp)
prov = f"{tmp}/sagemaker-resource-metadata.json"; shutil.copyfile(meta, prov)   # copia sin editar

with mlflow.start_run(experiment_id=exp.experiment_id, run_name="final_sentiment140") as run:
    mlflow.set_tags({"lab_run_type": "final", "lab_protocol_run_id": pid,
                     "lab_selected_experiment_run_id": sel.info.run_id,
                     "lab_configuration_id": t["lab_configuration_id"],
                     "lab_member_id": t["lab_member_id"], "notebook_arn": arn})
    mlflow.log_param("training_size", res["training_size"])
    mlflow.log_metric("test_macro_f1", res["test_macro_f1"])
    mlflow.log_artifact(cfg, "run")
    mlflow.log_artifact(prov, "provenance")
    mlflow.log_artifact(f"{H}/reports/error_analysis.csv", "reports")
    mlflow.log_artifact(f"{H}/reports/error_analysis.md", "reports")
    info = mlflow.pyfunc.log_model(
        name="model", python_model=model_def.SentimentModel(),
        artifacts={"pipeline": f"{H}/artifacts/final_pipeline.joblib"},
        code_paths=[f"{H}/src/model_def.py"],
        pip_requirements=[f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
                          f"pandas=={pandas.__version__}", f"numpy=={numpy.__version__}", f"joblib=={joblib.__version__}"],
        registered_model_name="sentiment140")
    final_id = run.info.run_id
    version = info.registered_model_version

c.set_registered_model_alias("sentiment140", "champion", str(version))
print("run final:", final_id, "| versión:", version, "| seleccionado:", sel.info.run_id, "| cfg:", t["lab_configuration_id"])

m = mlflow.pyfunc.load_model("models:/sentiment140@champion")
print(m.predict(["i loved it", "worst day ever", "not bad at all"]))
