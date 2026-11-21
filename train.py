import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from dataset import load_split, get_class_names
from features import prepare_features
from model import MyCNN, get_models

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train and evaluate linear classifiers")
    parser.add_argument(
        "--feature_type",
        type=str,
        default="color",
        choices=["pixels", "color"],
        help="type rapresentations of the features"
    )
    parser.add_argument(
        "--ls_lambda",
        type=float,
        default=10.0,
        help="regolarization parameter for LeastSquare"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="fast test on small set of samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="output directory for matrics and graphs"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="split used for evaluation: valid for tuning, test for final evaluation"
    )

    return parser.parse_args()

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path
):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=30)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



def main():
    args = parse_args()
    out_dir = Path(args.output_dir) / args.eval_split / args.feature_type
    out_dir.mkdir(parents=True, exist_ok=True)

    max_train = 150 if args.debug else None
    max_val = 50 if args.debug else None
    cw = None if args.class_weight == "none" else args.class_weight

    print(f"loading dataset (debug={args.debug})")
    x_train_crops, y_train = load_split(split_name="train", max_samples=max_train, streaming=args.debug)
    x_val_crops, y_val = load_split(split_name=args.eval_split, max_samples=max_val, streaming=args.debug)
    print(f"train crops: {len(y_train)} | val crops: {len(y_val)}")

    print(f"features extraction ({args.feature_type})")
    x_train, x_val, _ = prepare_features(x_train_crops, x_val_crops, mode=args.feature_type)
    print(f"shape x_train: {x_train.shape} | x_val: {x_val.shape}")

    print("evaluation of the models")
    models = get_models(ls_lambda=args.ls_lambda, svm_c=args.svm_c, class_weight=cw)
    if args.feature_type == "pixels":
        # pop the slow models
        for slow_model in ["Boosting"]:
            models.pop(slow_model, None)
    results = {}

    class_names = get_class_names()
    present_classes = np.unique(np.concatenate([y_train, y_val]))
    target_names=[]
    for i in present_classes:
        if 0 <= i < len(class_names):
            target_names.append(class_names[i])
        else:
            target_names.append(f"class_{i}")

    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(x_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = clf.predict(x_val)
        acc = accuracy_score(y_val, y_pred)
        train_acc = accuracy_score(y_train, clf.predict(x_train))
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_val, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )

        results[name] = {
            "accuracy": round(float(acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "train_time_sec": round(float(train_time), 3),
            "for_class": report
        }

        print(f"\n{name}")
        print(f"accuracy: {acc:.4f} | train acc: {train_acc:.4f} | F1: {macro_f1:.4f} | time: {train_time:.2f}s")
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

        save_confusion_matrix(
            y_val,
            y_pred,
            labels=target_names,
            title=f"{name} ({args.feature_type}) - normalized CM",
            save_path=out_dir / f"cm_{name.lower()}.png"
        )

    if args.feature_type == "color":
        cnn_epochs = 2 if args.debug else 15
        cnn = MyCNN(epochs=cnn_epochs, batch_size=128, lr=1e-3, class_weight=cw)
        t0 = time.perf_counter()
        cnn.fit(x_train_crops, y_train)
        train_time = time.perf_counter() - t0
        train_acc = accuracy_score(y_train, cnn.predict(x_train_crops))
        y_pred = cnn.predict(x_val_crops)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_val, y_pred, labels=present_classes, target_names=target_names, output_dict=True, zero_division=0
        )

        results["MyCNN"] = {
            "accuracy": round(float(acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "train_time_sec": round(float(train_time), 3),
            "for_class": report
        }

        print("\nMyCNN")
        print(f"accuracy: {acc:.4f} | train acc: {train_acc:.4f} | F1: {macro_f1:.4f} | time: {train_time:.2f}s")
        print(classification_report(y_val, y_pred, labels=present_classes, target_names=target_names, zero_division=0))

        save_confusion_matrix(
            y_val,
            y_pred,
            labels=target_names,
            title="MyCNN (raw_crops) - normalized CM",
            save_path=out_dir / "cm_mycnn.png"
        )

    print(f"saving results in {out_dir / 'metrics.json'}...")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()