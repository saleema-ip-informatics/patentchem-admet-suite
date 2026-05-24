import requests
import pandas as pd
import numpy as np
import os

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_egfr_bioactivity(limit=1000):
    """Fetch EGFR IC50 bioactivity data from ChEMBL REST API."""
    url = f"{BASE_URL}/activity.json"
    params = {
        "target_chembl_id": "CHEMBL203",
        "standard_type":    "IC50",
        "limit":            limit,
        "offset":           0,
    }
    response = requests.get(url, params=params)
    data = response.json()
    records = data["activities"]
    df = pd.DataFrame(records)
    return df


def save_raw_data(df, path="data/raw/egfr_ic50_raw.csv"):
    """Save raw dataframe to CSV, creating folders if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Raw data saved to {path}")


def clean_bioactivity_data(df):
    """
    Clean raw ChEMBL bioactivity data.
    - Keeps essential columns
    - Removes missing SMILES or activity values
    - Converts IC50 (nM) to pIC50
    - Clips to realistic drug range (3–12)
    - Adds binary activity label (pIC50 >= 6 = active)
    """
    cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value',
            'standard_units', 'standard_type', 'pchembl_value']
    df = df[cols].copy()

    # Remove rows without SMILES or activity
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])

    # Filter out zero/negative values before log transform
    df = df[df['standard_value'] > 0]

    # Convert IC50 (nM) to pIC50
    df['pIC50'] = -np.log10(df['standard_value'].astype(float) * 1e-9)

    # Clip to realistic drug range
    df = df[(df['pIC50'] >= 3) & (df['pIC50'] <= 12)]

    # Binary activity label: pIC50 >= 6 = active (IC50 <= 1 uM)
    df['active'] = (df['pIC50'] >= 6).astype(int)

    print(f"Clean dataset: {len(df)} compounds")
    print(f"Active: {df['active'].sum()} | Inactive: {(df['active']==0).sum()}")

    return df


def load_clean_data(raw_path="data/raw/egfr_ic50_raw.csv"):
    """Load raw CSV and return cleaned dataframe."""
    df = pd.read_csv(raw_path)
    return clean_bioactivity_data(df)
