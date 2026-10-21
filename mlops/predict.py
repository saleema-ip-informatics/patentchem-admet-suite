import os
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def predict(model_path, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Enter a valid, non-empty molecular SMILES.")

    bundle = joblib.load(model_path)
    settings = bundle["fingerprint"]

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=settings["radius"],
        fpSize=settings["n_bits"],
    )
    X = generator.GetFingerprintAsNumPy(mol).reshape(1, -1)
    value = float(bundle["model"].predict(X)[0])

    if not np.isfinite(value):
        raise ValueError("Model returned a non-finite prediction.")


    import pandas as pd
    from rdkit import DataStructs

    assignments = pd.read_csv(
        Path(model_path).parent / "split_assignments.csv"
    )
    training = assignments.loc[assignments["split"] == "train"]

    query_fp = generator.GetFingerprint(mol)
    similarities = []
    for smiles in training["structure_id"]:
        reference = Chem.MolFromSmiles(smiles)
        if reference is None:
            raise ValueError("Invalid structure in training records.")
        similarities.append(
            DataStructs.TanimotoSimilarity(
                query_fp, generator.GetFingerprint(reference)
            )
        )
    if not similarities:
        raise ValueError("No training structures available.")

    nearest_similarity = max(similarities)

    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "predicted_pIC50": value,
        "nearest_training_similarity": nearest_similarity,
        "similarity_note": (
            "Morgan fingerprint Tanimoto similarity; "
            "not a calibrated confidence score."
        ),
        "assay_id": bundle["assay_id"],
        "run_folder": os.environ.get("MODEL_RUN_ID", Path(model_path).parent.name),
        "limitation": (
            "Learning model trained on 38 molecules. "
            "Applicability domain not assessed; "
            "not evidence of biological activity."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()

    try:
        result = predict(args.model, args.smiles)
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(result, indent=2))
