import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def smiles_to_mol(smiles):
    """Parse SMILES string, return RDKit mol object or None if invalid."""
    return Chem.MolFromSmiles(smiles)


def compute_lipinski_descriptors(mol):
    """
    Compute Lipinski Rule of Five descriptors.
    Returns a dictionary with 8 physicochemical properties.
    """
    return {
        'MW':           Descriptors.MolWt(mol),
        'LogP':         Descriptors.MolLogP(mol),
        'HBD':          rdMolDescriptors.CalcNumHBD(mol),
        'HBA':          rdMolDescriptors.CalcNumHBA(mol),
        'TPSA':         Descriptors.TPSA(mol),
        'RotBonds':     rdMolDescriptors.CalcNumRotatableBonds(mol),
        'RingCount':    rdMolDescriptors.CalcNumRings(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def compute_morgan_fp(mol, radius=2, n_bits=1024):
    """
    Compute Morgan (ECFP4) fingerprint as numpy array.
    Uses MorganGenerator (modern RDKit API, no deprecation warnings).
    """
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.array(fp)


def compute_maccs_fp(mol):
    """Compute MACCS 166-bit structural keys as numpy array."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def featurize_dataframe(df, smiles_col='canonical_smiles'):
    """
    Full featurization pipeline for a dataframe of SMILES.

    Steps:
    - Parse SMILES to RDKit mol objects
    - Filter out invalid SMILES
    - Compute Lipinski descriptors
    - Compute Morgan fingerprints (1024-bit)
    - Compute MACCS keys (166-bit)
    - Build combined feature matrix (descriptors + Morgan FP)

    Returns:
        df_clean     : cleaned dataframe (invalid SMILES removed)
        X_combined   : numpy array of shape (n_compounds, 1032)
        X_descriptors: numpy array of shape (n_compounds, 8)
        morgan_fp    : numpy array of shape (n_compounds, 1024)
        maccs_fp     : numpy array of shape (n_compounds, 166)
        desc_df      : DataFrame of Lipinski descriptors
        feature_names: list of all feature names
    """
    # Parse SMILES
    mols = df[smiles_col].apply(smiles_to_mol)
    valid_mask = mols.notna()
    df_clean = df[valid_mask].copy()
    mols = mols[valid_mask]

    print(f"Valid molecules: {len(df_clean)} / {len(df)} total")

    # Lipinski descriptors
    desc_df = pd.DataFrame(
        [compute_lipinski_descriptors(m) for m in mols]
    )
    desc_df.index = df_clean.index

    # Morgan fingerprints (1024 bits)
    morgan_fp = np.array([compute_morgan_fp(m) for m in mols])

    # MACCS keys (166 bits)
    maccs_fp = np.array([compute_maccs_fp(m) for m in mols])

    # Combined feature matrix: descriptors + Morgan FP
    X_descriptors = desc_df.values
    X_combined = np.hstack([X_descriptors, morgan_fp])

    # Feature names for interpretability
    feature_names = list(desc_df.columns) + [
        f"Morgan_bit_{i}" for i in range(morgan_fp.shape[1])
    ]

    print(f"Feature matrix shape: {X_combined.shape}")
    print(f"Descriptor names: {list(desc_df.columns)}")

    return df_clean, X_combined, X_descriptors, morgan_fp, maccs_fp, desc_df, feature_names
