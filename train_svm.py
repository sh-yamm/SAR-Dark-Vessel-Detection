"""
Train the SVM ship classifier on preprocessed SAR chips.

Usage:
    py -3.11 train_svm.py --config config/config.yaml
    py -3.11 train_svm.py --config config/config.yaml --out checkpoints/svm_model.pkl
"""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from models.svm_detector import extract_features


def load_chips(chips_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all chips from a split directory.

    Positive samples: files starting with 'pos_' (contain ships)
    Negative samples: files starting with 'neg_' (background only)

    Returns X (N, F) feature matrix and y (N,) labels.
    """
    X, y = [], []
    root = chips_root / split

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for img_path in sorted(scene_dir.glob("*_img.npy")):
            chip  = np.load(str(img_path))    # (2, H, W)
            label = 1 if img_path.stem.startswith("pos") else 0
            X.append(extract_features(chip))
            y.append(label)

    return np.stack(X), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out",    default="checkpoints/svm_model.pkl")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    chips_root = Path(cfg["data"]["chips_dir"])

    print("Loading train chips ...")
    X_train, y_train = load_chips(chips_root, "train")
    print(f"  Train: {X_train.shape[0]} chips  "
          f"({y_train.sum()} positive, {(y_train==0).sum()} negative)")

    print("Loading val chips ...")
    X_val, y_val = load_chips(chips_root, "val")
    print(f"  Val:   {X_val.shape[0]} chips  "
          f"({y_val.sum()} positive, {(y_val==0).sum()} negative)")

    print("\nTraining LinearSVC ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    print("\nValidation results:")
    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred,
                                 target_names=["background", "ship"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
