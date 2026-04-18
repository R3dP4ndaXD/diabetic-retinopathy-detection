from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.utils import generate_run_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an XGBoost meta-learner from OOF probability CSVs."
    )
    p.add_argument(
        "--oof-csvs",
        nargs="+",
        required=True,
        help="One OOF CSV per base model (must contain image_path, label, prob_* and fold).",
    )
    p.add_argument(
        "--test-prob-csvs",
        nargs="+",
        default=None,
        help="Optional test probability CSVs (same order/count as --oof-csvs).",
    )
    p.add_argument("--output-dir", default="artifacts/meta_xgb")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--xgb-n-estimators", type=int, default=500)
    p.add_argument("--xgb-max-depth", type=int, default=4)
    p.add_argument("--xgb-learning-rate", type=float, default=0.03)
    p.add_argument("--xgb-subsample", type=float, default=0.9)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    p.add_argument("--xgb-tree-method", default="hist")
    p.add_argument("--n-jobs", type=int, default=8)

    return p.parse_args()


def _prob_columns(df: pd.DataFrame) -> list[str]:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("Expected probability columns named prob_0, prob_1, ...")

    def key(name: str) -> int:
        try:
            return int(name.split("_")[1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid probability column name: {name}") from exc

    return sorted(prob_cols, key=key)


def _load_and_merge_prob_csvs(csv_paths: list[Path], require_fold: bool) -> tuple[pd.DataFrame, list[str], int]:
    merged: pd.DataFrame | None = None
    feature_cols: list[str] = []
    num_classes: int | None = None

    for i, path in enumerate(csv_paths):
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        required = {"image_path", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        if require_fold and "fold" not in df.columns:
            raise ValueError(f"{path} must contain a 'fold' column for OOF CV training.")

        probs = _prob_columns(df)
        if num_classes is None:
            num_classes = len(probs)
        elif len(probs) != num_classes:
            raise ValueError("All input CSVs must have the same number of prob_* columns.")

        rename = {c: f"m{i}_{c}" for c in probs}
        cols = ["image_path", "label"] + (["fold"] if "fold" in df.columns else []) + probs
        part = df[cols].rename(columns=rename)

        if merged is None:
            merged = part
            feature_cols.extend(rename[c] for c in probs)
        else:
            part = part[["image_path", "label"] + [rename[c] for c in probs]]
            before = len(merged)
            merged = merged.merge(part, on=["image_path", "label"], how="inner", validate="one_to_one")
            if len(merged) != before:
                raise ValueError(
                    "CSV alignment mismatch while merging OOF/test probability files. "
                    "Ensure all files have identical image_path + label rows."
                )
            feature_cols.extend(rename[c] for c in probs)

    assert merged is not None
    assert num_classes is not None
    return merged, feature_cols, num_classes


def _build_xgb(args: argparse.Namespace, num_classes: int):
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "xgboost is not installed. Add it to requirements and install dependencies."
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_lambda=args.xgb_reg_lambda,
        random_state=args.seed,
        eval_metric="mlogloss",
        early_stopping_rounds=20,   # stop if mlogloss doesn't improve for 20 rounds
        tree_method=args.xgb_tree_method,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    args = _parse_args()
    run_id = generate_run_id()

    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_paths = [Path(p) for p in args.oof_csvs]
    train_df, feature_cols, num_classes = _load_and_merge_prob_csvs(oof_paths, require_fold=True)

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int64)
    folds = train_df["fold"].to_numpy(dtype=np.int64)

    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("Need at least 2 distinct fold values in OOF CSVs.")

    meta_oof_pred = np.full(len(train_df), -1, dtype=np.int64)

    print(f"Run ID                : {run_id}")
    print(f"Training rows         : {len(train_df)}")
    print(f"Base members          : {len(oof_paths)}")
    print(f"Meta feature dim      : {X.shape[1]}")
    print(f"Classes               : {num_classes}")

    fold_scores: list[float] = []
    for fold in unique_folds:
        tr = folds != fold
        va = folds == fold

        model = _build_xgb(args, num_classes)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

        pred = model.predict(X[va]).astype(np.int64)
        meta_oof_pred[va] = pred
        fold_kappa = cohen_kappa_score(y[va], pred, weights="quadratic")
        fold_scores.append(float(fold_kappa))
        print(f"Fold {int(fold)} meta QWK: {fold_kappa:.4f}")

    if np.any(meta_oof_pred < 0):
        raise RuntimeError("Some OOF rows did not receive meta predictions.")

    meta_oof_kappa = cohen_kappa_score(y, meta_oof_pred, weights="quadratic")
    meta_oof_acc = accuracy_score(y, meta_oof_pred)
    meta_oof_f1 = f1_score(y, meta_oof_pred, average="macro", zero_division=0)

    print(f"\nMeta OOF QWK : {meta_oof_kappa:.4f}")
    print(f"Meta OOF ACC : {meta_oof_acc:.4f}")
    print(f"Meta OOF F1m : {meta_oof_f1:.4f}")

    train_out = train_df[["image_path", "label", "fold"]].copy()
    train_out["meta_pred_label"] = meta_oof_pred
    train_out.to_csv(out_dir / "meta_oof_predictions.csv", index=False)

    # Final model: train on all OOF data, use mean best iteration from CV folds
    final_model = _build_xgb(args, num_classes)
    final_model.set_params(early_stopping_rounds=None)
    final_model.fit(X, y, verbose=False)

    model_path = out_dir / "meta_xgb_model.json"
    final_model.save_model(model_path)

    test_summary: dict[str, float | str | None] = {
        "test_qwk": None,
        "test_acc": None,
        "test_f1_macro": None,
        "test_predictions_csv": None,
    }

    if args.test_prob_csvs is not None:
        test_paths = [Path(p) for p in args.test_prob_csvs]
        if len(test_paths) != len(oof_paths):
            raise ValueError(
                "--test-prob-csvs must have the same number of files as --oof-csvs."
            )

        test_df, test_features, _ = _load_and_merge_prob_csvs(test_paths, require_fold=False)
        if test_features != feature_cols:
            raise ValueError(
                "Feature column mismatch between OOF CSVs and test probability CSVs."
            )

        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        test_pred = final_model.predict(X_test).astype(np.int64)
        test_proba = final_model.predict_proba(X_test)

        test_out = test_df[["image_path", "label"]].copy()
        for c in range(num_classes):
            test_out[f"meta_prob_{c}"] = test_proba[:, c]
        test_out["meta_pred_label"] = test_pred

        test_predictions_csv = out_dir / "meta_test_predictions.csv"
        test_out.to_csv(test_predictions_csv, index=False)
        test_summary["test_predictions_csv"] = str(test_predictions_csv)

        if "label" in test_df.columns:
            y_test = test_df["label"].to_numpy(dtype=np.int64)
            test_qwk = cohen_kappa_score(y_test, test_pred, weights="quadratic")
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
            test_summary["test_qwk"] = float(test_qwk)
            test_summary["test_acc"] = float(test_acc)
            test_summary["test_f1_macro"] = float(test_f1)

            print(f"\nMeta TEST QWK : {test_qwk:.4f}")
            print(f"Meta TEST ACC : {test_acc:.4f}")
            print(f"Meta TEST F1m : {test_f1:.4f}")

    summary = {
        "run_id": run_id,
        "num_members": len(oof_paths),
        "num_classes": num_classes,
        "feature_dim": int(X.shape[1]),
        "meta_oof_qwk": float(meta_oof_kappa),
        "meta_oof_acc": float(meta_oof_acc),
        "meta_oof_f1_macro": float(meta_oof_f1),
        "fold_qwk": fold_scores,
        "model_path": str(model_path),
        "feature_columns": feature_cols,
        **test_summary,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model   : {model_path}")
    print(f"Saved metrics : {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
