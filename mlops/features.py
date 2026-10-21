import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=1024
)


def build_features(data):
    data = data.copy()
    fingerprints = []
    identities = []
    scaffolds = []

    for index, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}")

        identities.append(
            Chem.MolToSmiles(mol, isomericSmiles=True)
        )
        scaffolds.append(
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        )
        fingerprints.append(
            GENERATOR.GetFingerprintAsNumPy(mol)
        )

    data["structure_id"] = identities
    data["scaffold"] = scaffolds

    if data["structure_id"].duplicated().any():
        raise ValueError("Repeated structures need review before splitting.")

    return data, np.asarray(fingerprints, dtype=np.uint8)


if __name__ == "__main__":
    from train import load_assay_data

    data, features = build_features(load_assay_data())
    print("Feature matrix:", features.shape)
    print("Unique structures:", data["structure_id"].nunique())
    print("Scaffold groups:", data["scaffold"].nunique())
