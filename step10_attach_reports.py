"""Adjunta reports/error_analysis.csv y .md como artefactos del run final."""
import mlflow
from common import set_tracking

set_tracking()

with open(".final_run_id") as fh:
    final_run_id = fh.read().strip()

client = mlflow.MlflowClient()
client.log_artifact(final_run_id, "reports/error_analysis.csv", artifact_path="reports")
client.log_artifact(final_run_id, "reports/error_analysis.md", artifact_path="reports")

print(f"Artefactos adjuntados al run final: {final_run_id}")

# Verificación
artifacts = client.list_artifacts(final_run_id, "reports")
for a in artifacts:
    print(" -", a.path)
