import os
import json
import urllib.request
import urllib.error
import pytest

URL = os.environ.get("API_URL", "http://127.0.0.1:8001/predict")


def post(smiles):
    request = urllib.request.Request(
        URL,
        data=json.dumps({"smiles": smiles}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error

    with response:
        return response.code, json.load(response)


def test_valid_smiles():
    status, body = post("CCO")
    assert status == 200
    assert body["canonical_smiles"] == "CCO"
    expected_run = os.environ.get("EXPECTED_RUN_ID")
    assert expected_run, "Set EXPECTED_RUN_ID before running API tests."
    assert body["run_folder"] == expected_run
    assert body["predicted_pIC50"] == pytest.approx(
        7.545550809809083, abs=1e-10
    )
    assert body["nearest_training_similarity"] == pytest.approx(
        0.13636363636363635
    )


def test_invalid_smiles():
    status, body = post("invalid")
    assert status == 422
    assert body["detail"] == "Enter a valid, non-empty molecular SMILES."


def test_health_endpoint():
    health_url = os.environ.get("HEALTH_URL")
    if not health_url:
        pytest.skip("Health check applies to the SageMaker adapter.")
    with urllib.request.urlopen(health_url, timeout=30) as response:
        assert response.status == 200
