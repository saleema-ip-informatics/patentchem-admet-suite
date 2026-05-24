import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold

# ─── EGFR Pharmacophore SMARTS ────────────────────────────────────────────────
# These patterns reflect the key structural features of EGFR inhibitor patents
# Erlotinib, Gefitinib, Afatinib, Osimertinib all share quinazoline core
PHARMACOPHORES = {
    'quinazoline_core':   'c1cnc2ccccc2n1',   # core of Iressa, Tagrisso, Erlotinib
    'aniline_rgroup':     'c1ccc(N)cc1',       # common R-group attachment point
    'pyrimidine_core':    'c1ncnc(N)n1',       # alternative core scaffold
    'acrylamide_warhead': 'C=CC(=O)N',         # covalent warhead (Afatinib, Osimertinib)
}

QUINAZOLINE_CORE_SMARTS = '[#6]1:[#7]:[#6]:[#7]:[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:1:2'


def get_murcko_scaffold(smiles):
    """
    Extract Bemis-Murcko scaffold from a SMILES string.
    In patent terms: the 'core' of a Markush general formula.
    Returns scaffold SMILES or None if invalid.
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def get_top_scaffolds(df, smiles_col='canonical_smiles', top_n=10):
    """
    Extract Murcko scaffolds for all compounds and return top N by frequency.
    Scaffold frequency is a proxy for patent claim breadth.
    """
    df = df.dropna(subset=[smiles_col]).copy()
    df[smiles_col] = df[smiles_col].astype(str)
    df['murcko_scaffold'] = df[smiles_col].apply(get_murcko_scaffold)

    scaffold_freq = df['murcko_scaffold'].value_counts().reset_index()
    scaffold_freq.columns = ['scaffold', 'compound_count']

    print(f"Unique scaffolds found: {len(scaffold_freq)}")
    print(f"Top {top_n} scaffolds (Markush core families):")
    print(scaffold_freq.head(top_n))

    return df, scaffold_freq


def detect_pharmacophore_zones(smiles, pharmacophores=PHARMACOPHORES):
    """
    Check which pharmacophore patterns are present in a molecule.
    Analogous to identifying R-group positions in a Markush formula.
    Returns a dict of {pattern_name: True/False}.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: False for name in pharmacophores}
    results = {}
    for name, smarts in pharmacophores.items():
        pattern = Chem.MolFromSmarts(smarts)
        results[name] = mol.HasSubstructMatch(pattern) if pattern else False
    return results


def add_pharmacophore_flags(df, smiles_col='canonical_smiles'):
    """
    Add pharmacophore presence columns to the dataframe.
    Each column is True/False for that structural pattern.
    """
    pharm_df = pd.DataFrame(
        [detect_pharmacophore_zones(s) for s in df[smiles_col]],
        index=df.index
    )
    return pd.concat([df, pharm_df], axis=1)


def rgroup_decompose(smiles_list, core_smarts=QUINAZOLINE_CORE_SMARTS):
    """
    Decompose a compound series against a Markush core using RDKit.
    Returns R-group SMILES for each position — exactly what patent analysts do.

    Args:
        smiles_list : list of SMILES strings
        core_smarts : SMARTS string for the Markush core scaffold

    Returns:
        rgroup_df : DataFrame with R-group SMILES per position
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    core = Chem.MolFromSmarts(core_smarts)

    groups, unmatched = rdRGroupDecomposition.RGroupDecompose(
        [core], mols, asSmiles=True, asRows=True
    )
    rgroup_df = pd.DataFrame(groups)
    print(f"Decomposed {len(rgroup_df)} compounds | Unmatched: {len(unmatched)}")
    print(f"R-group positions identified: {list(rgroup_df.columns)}")
    return rgroup_df


def run_markush_analysis(df, smiles_col='canonical_smiles',
                         save_path='data/processed/egfr_markush_processed.csv'):
    """
    Full Markush analysis pipeline:
    1. Murcko scaffold extraction + frequency ranking
    2. Pharmacophore zone detection
    3. R-group decomposition on quinazoline subset

    Returns:
        df_out     : dataframe with scaffold + pharmacophore columns added
        scaffold_freq : scaffold frequency table
        rgroup_df  : R-group decomposition results (quinazoline subset)
    """
    # Step 1 — Murcko scaffolds
    df_out, scaffold_freq = get_top_scaffolds(df, smiles_col=smiles_col)

    # Step 2 — Pharmacophore flags
    df_out = add_pharmacophore_flags(df_out, smiles_col=smiles_col)

    # Step 3 — R-group decomposition on quinazoline compounds
    quinazoline_smiles = df_out[
        df_out['quinazoline_core'] == True
    ][smiles_col].head(100).tolist()

    rgroup_df = None
    if len(quinazoline_smiles) > 5:
        rgroup_df = rgroup_decompose(quinazoline_smiles)

    # Save
    df_out.to_csv(save_path, index=False)
    print(f"Markush analysis saved to {save_path}")

    return df_out, scaffold_freq, rgroup_df
