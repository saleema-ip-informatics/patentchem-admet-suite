from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/egfr_ic50_raw.csv"
ASSAY_ID = "CHEMBL679944"


def load_assay_data():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    values = pd.to_numeric(df["standard_value"], errors="coerce")

    mask = (
        df["assay_chembl_id"].eq(ASSAY_ID)
        & df["standard_type"].eq("IC50")
        & df["standard_units"].eq("nM")
        & df["standard_relation"].eq("=")
        & df["bao_label"].eq("single protein format")
        & df["data_validity_comment"].str.strip().eq("")
        & df["potential_duplicate"].ne("1")
        & df["canonical_smiles"].str.strip().ne("")
        & values.gt(0)
        & np.isfinite(values)
    )

    selected = df.loc[mask].copy()
    selected["IC50_nM"] = values.loc[mask]
    selected["pIC50"] = 9 - np.log10(selected["IC50_nM"])

    if selected.empty:
        raise ValueError("No eligible assay records found.")

    return selected


if __name__ == "__main__":
    data = load_assay_data()
    print("Assay:", ASSAY_ID)
    print("Eligible records:", len(data))
    print(data[["molecule_chembl_id", "IC50_nM", "pIC50"]].head())
