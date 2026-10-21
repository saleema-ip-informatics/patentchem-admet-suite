from sklearn.model_selection import GroupShuffleSplit
from train import load_assay_data
from features import build_features


def split_by_scaffold(data, features):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(features, groups=data["scaffold"])
    )

    train_scaffolds = set(data.iloc[train_idx]["scaffold"])
    test_scaffolds = set(data.iloc[test_idx]["scaffold"])

    if train_scaffolds & test_scaffolds:
        raise ValueError("Scaffold overlap detected.")

    return train_idx, test_idx


if __name__ == "__main__":
    data, features = build_features(load_assay_data())
    train_idx, test_idx = split_by_scaffold(data, features)

    print("Training molecules:", len(train_idx))
    print("Testing molecules:", len(test_idx))
    print("Training scaffolds:",
          data.iloc[train_idx]["scaffold"].nunique())
    print("Testing scaffolds:",
          data.iloc[test_idx]["scaffold"].nunique())
    print("Scaffold overlap: 0")
