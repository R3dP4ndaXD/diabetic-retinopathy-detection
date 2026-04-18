import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _load_csv(data_dir: str, csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image"].apply(lambda x: os.path.join(data_dir, f"{x}.jpeg"))
    df = df.rename(columns={"level": "label"})
    return df


def _assert_all_image_paths_exist(df: pd.DataFrame, csv_path: str, split_name: str) -> pd.DataFrame:
    missing_mask = ~df["image_path"].apply(os.path.exists)
    if not missing_mask.any():
        return df

    missing_paths = df.loc[missing_mask, "image_path"].tolist()
    sample_size = min(10, len(missing_paths))
    sample = "\n".join(f"  - {path}" for path in missing_paths[:sample_size])
    raise FileNotFoundError(
        f"{split_name}: found {len(missing_paths)} missing image files referenced by {csv_path}. "
        "No rows were dropped.\n"
        f"First {sample_size} missing paths:\n{sample}"
    )


def _patient_id_from_image_path(image_path: str) -> tuple[str, bool]:
    """
    Extract patient id from EyePACS-like filenames:
      - <patient_id>_left.jpeg
      - <patient_id>_right.jpeg
    Falls back to the full stem when the pattern is not present; the second
    return value is True on a fallback so callers can log the rate.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if stem.endswith("_left") or stem.endswith("_right"):
        return stem.rsplit("_", 1)[0], False
    return stem, True


def load_train_data(data_dir: str, csv_path: str) -> pd.DataFrame:
    df = _load_csv(data_dir, csv_path)
    df = _assert_all_image_paths_exist(df, csv_path=csv_path, split_name="train")
    df = df.reset_index(drop=True)
    return df[["image_path", "label"]]


def load_test_data(data_dir: str, csv_path: str, usage: str | None = None) -> pd.DataFrame:
    df = _load_csv(data_dir, csv_path)
    if usage and "Usage" in df.columns:
        df = df[df["Usage"].str.strip() == usage]
    df = _assert_all_image_paths_exist(df, csv_path=csv_path, split_name="test")
    df = df.reset_index(drop=True)
    return df[["image_path", "label"]]


def main(
    data_dir: str,
    csv_path: str,
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str,
    test_labels_csv: str,
    test_data_dir: str | None = None,
    test_usage: str | None = None,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    df_train_all = load_train_data(data_dir, csv_path)
    df_test = load_test_data(test_data_dir or data_dir, test_labels_csv, usage=test_usage)

    labels = df_train_all["label"].to_numpy()
    parsed = df_train_all["image_path"].apply(_patient_id_from_image_path)
    groups = parsed.apply(lambda t: t[0]).to_numpy()
    fallback_count = int(parsed.apply(lambda t: t[1]).sum())
    fallback_fraction = fallback_count / len(df_train_all)
    if fallback_count:
        print(
            f"[split_dataset] Patient-id fallback (stem == group) used for "
            f"{fallback_count}/{len(df_train_all)} rows "
            f"({fallback_fraction:.2%}). Non-EyePACS filenames will be treated "
            "as one patient per image."
        )
    else:
        print("[split_dataset] All filenames matched _left/_right pattern.")

    n_splits = max(2, int(round(1.0 / val_size)))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_idx, val_idx = next(splitter.split(df_train_all, labels, groups=groups))
    df_train = df_train_all.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_all.iloc[val_idx].reset_index(drop=True)

    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = train_groups & val_groups
    if overlap:
        sample_overlap = list(sorted(overlap))[:10]
        raise RuntimeError(
            "Patient-level leakage detected between train and val groups. "
            f"First overlapping patient IDs: {sample_overlap}"
        )

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    actual_val_fraction = len(df_val) / len(df_train_all)
    print(
        f"Train: {len(df_train)}, Val: {len(df_val)} "
        f"(val_fraction={actual_val_fraction:.4f}), Test (official): {len(df_test)}"
    )
    if test_usage:
        print(f"  Test filtered to Usage='{test_usage}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build train/val/test CSVs for the DR dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/diabetic-retinopathy-dataset/resized/train",
        help="Directory containing the preprocessed train images.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/diabetic-retinopathy-dataset/trainLabels.csv",
        help="Kaggle trainLabels.csv (columns: image, level).",
    )
    parser.add_argument(
        "--train-csv-path",
        type=str,
        default="data/diabetic-retinopathy-dataset/train.csv",
    )
    parser.add_argument(
        "--val-csv-path",
        type=str,
        default="data/diabetic-retinopathy-dataset/val.csv",
    )
    parser.add_argument(
        "--test-csv-path",
        type=str,
        default="data/diabetic-retinopathy-dataset/test.csv",
    )
    parser.add_argument(
        "--test-labels-csv",
        type=str,
        required=True,
        help="Path to testLabels.csv (columns: image, level, Usage).",
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=None,
        help="Directory containing the preprocessed test images (defaults to --data-dir).",
    )
    parser.add_argument(
        "--test-usage",
        type=str,
        choices=["Public", "Private"],
        default=None,
        help="Keep only Public or Private rows from testLabels.csv (default: keep all).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of the train set to use for validation.",
    )
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        train_csv_path=args.train_csv_path,
        val_csv_path=args.val_csv_path,
        test_csv_path=args.test_csv_path,
        test_labels_csv=args.test_labels_csv,
        test_data_dir=args.test_data_dir,
        test_usage=args.test_usage,
        val_size=args.val_size,
        random_state=args.random_state,
    )
