from pathlib import Path
import joblib
import numpy as np

from train import load_assay_data
from features import build_features
from split import split_by_scaffold


def test_model_round_trip(tmp_path):
    import os

    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError("Set MODEL_PATH to the model being tested.")
    artifact = Path(model_path)
    bundle = joblib.load(artifact)

    data, X = build_features(load_assay_data())
    _, test_idx = split_by_scaffold(data, X)

    original = bundle["model"].predict(X[test_idx])

    copy_path = tmp_path / "model_copy.joblib"
    joblib.dump(bundle, copy_path)
    restored = joblib.load(copy_path)
    reloaded = restored["model"].predict(X[test_idx])

    assert np.isfinite(reloaded).all()
    np.testing.assert_allclose(
        original, reloaded, rtol=1e-12, atol=1e-12
    )


def test_source_data_matches_manifest():
    import os
    import json
    import hashlib

    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError("Set MODEL_PATH to the model being tested.")

    run_dir = Path(model_path).parent
    manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )

    root = Path(__file__).resolve().parents[1]
    source = root / manifest["source_file"]
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert actual_hash == manifest["sha256"], (
        "Current CSV differs from this run's recorded training source."
    )
