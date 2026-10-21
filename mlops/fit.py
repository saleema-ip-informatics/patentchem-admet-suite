import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train import load_assay_data
from features import build_features
from split import split_by_scaffold


def run_training():
    data, X = build_features(load_assay_data())
    train_idx, test_idx = split_by_scaffold(data, X)
    y = data["pIC50"].to_numpy()

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    print("Train:", len(train_idx), "| Test:", len(test_idx))

    metrics = {}

    for name, model in models.items():
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])

        mae = mean_absolute_error(y[test_idx], predictions)
        rmse = np.sqrt(
            mean_squared_error(y[test_idx], predictions)
        )
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "units": "pIC50"
        }



    from pathlib import Path
    import joblib

    from datetime import datetime, timezone

    artifacts = Path(__file__).resolve().parent / "artifacts"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = artifacts / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)

    joblib.dump(
        {
            "model": models["Random Forest"],
            "assay_id": "CHEMBL679944",
            "target": "pIC50",
            "fingerprint": {"radius": 2, "n_bits": 1024},
        },
        output / "model.joblib",
    )
    print("Saved:", output / "model.joblib")


    import hashlib
    import json
    from train import DATA_PATH

    manifest = {
        "source_file": "data/processed/egfr_ic50_raw.csv",
        "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "assay_id": "CHEMBL679944",
        "target": "pIC50",
        "training_records": len(train_idx),
        "testing_records": len(test_idx),
        "split_method": "scaffold_group_shuffle",
        "random_state": 42,
        "purpose": "MLOps learning demonstration"
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    assignments = data[
        ["molecule_chembl_id", "structure_id", "scaffold", "pIC50"]
    ].copy()
    assignments["split"] = "train"
    assignments.iloc[
        test_idx, assignments.columns.get_loc("split")
    ] = "test"
    assignments.to_csv(output / "split_assignments.csv", index=False)
    print("Saved data manifest and split assignments.")


    (output / "metrics.json").write_text(
        json.dumps({
            "models": metrics,
            "test_records": len(test_idx),
            "limitation": "Small holdout; learning demonstration only."
        }, indent=2),
        encoding="utf-8"
    )
    print("Saved evaluation metrics.")


    import platform
    from importlib.metadata import version

    environment = {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in [
                "rdkit", "numpy", "pandas",
                "scikit-learn", "joblib"
            ]
        }
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print("Saved environment versions.")

    import subprocess
    import sys

    tracking_script = Path(__file__).resolve().parent / "track_run.py"
    subprocess.run(
        [sys.executable, str(tracking_script), str(output)],
        check=True,
    )


    print("Metrics are in pIC50 units; learning demonstration only.")


if __name__ == "__main__":
    run_training()
