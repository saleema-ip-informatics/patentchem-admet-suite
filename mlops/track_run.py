from pathlib import Path
import argparse
import json
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("run_folder")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
folder = Path(args.run_folder).resolve()

manifest = json.loads(
    (folder / "data_manifest.json").read_text()
)
metrics = json.loads(
    (folder / "metrics.json").read_text()
)

mlflow.set_tracking_uri(
    "sqlite:///" + (root / "mlflow.db").as_posix()
)
mlflow.set_experiment("EGFR-assay-learning")

with mlflow.start_run(run_name=folder.name) as run:
    mlflow.log_params({
        "assay_id": manifest["assay_id"],
        "data_sha256": manifest["sha256"],
        "split_method": manifest["split_method"],
        "random_state": manifest["random_state"],
        "training_records": manifest["training_records"],
        "testing_records": manifest["testing_records"],
    })

    for name, values in metrics["models"].items():
        prefix = name.lower().replace(" ", "_")
        mlflow.log_metrics({
            f"{prefix}_mae": values["MAE"],
            f"{prefix}_rmse": values["RMSE"],
        })

    mlflow.set_tag("scope", "Learning demonstration; small holdout")
    mlflow.log_artifacts(str(folder), artifact_path="training_run")
    print("MLflow run ID:", run.info.run_id)
