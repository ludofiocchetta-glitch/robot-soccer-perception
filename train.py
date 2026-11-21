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
import torch

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
        default="both",
        choices=["valid", "test", "both"],
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

def get_target_names(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    present_classes = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [
        class_names[i] if 0 <= i < len(class_names) else f"class_{i}"
        for i in present_classes
    ]
    return present_classes, target_names


def main():
    args = parse_args()
    base_out = Path(args.output_dir) 
    base_out.mkdir(parents=True, exist_ok=True)

    max_train = 150 if args.debug else None
    max_val = 50 if args.debug else None
    cw = "balanced"

    splits_to_eval = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]

    # loading dataset
    print(f"loading dataset (debug={args.debug})")
    x_train_crops, y_train = load_split(split_name="train", max_samples=max_train, streaming=args.debug)
    eval_data = {}
    for sp in splits_to_eval:
        eval_data[sp] = load_split(split_name=sp, max_samples=max_val, streaming=args.debug)

    # feature extraction
    print(f"features extraction ({args.feature_type})")
    first_sp = splits_to_eval[0]
    x_train, first_x_eval, scaler = prepare_features(
        x_train_crops, eval_data[first_sp][0], mode=args.feature_type
    )
    eval_features = {first_sp: (first_x_eval, eval_data[first_sp][1], eval_data[first_sp][0])}

    if len(splits_to_eval) > 1:
        from features import extract_color_features, extract_pixels
        second_sp = splits_to_eval[1]
        raw_second = (
            extract_color_features(eval_data[second_sp][0])
            if args.feature_type == "color"
            else extract_pixels(eval_data[second_sp][0])
        )
        eval_features[second_sp] = (
            scaler.transform(raw_second),
            eval_data[second_sp][1],
            eval_data[second_sp][0],
        )
    # prepare result containers for split
    results_by_split = {sp: {} for sp in splits_to_eval}
    class_names = get_class_names()
    # train classical models once and evaluate on each requested split
    print("\ntraining and evaluating classical models...")
    models = get_models(ls_lambda=args.ls_lambda, class_weight=cw)
    if args.feature_type == "pixels":
        models.pop("Boosting", None)

    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(x_train, y_train)
        train_time = time.perf_counter() - t0
        train_acc = accuracy_score(y_train, clf.predict(x_train))

        for sp, (x_ev, y_ev, _) in eval_features.items():
            out_dir = base_out / sp / args.feature_type
            out_dir.mkdir(parents=True, exist_ok=True)

            y_pred = clf.predict(x_ev)
            acc = accuracy_score(y_ev, y_pred)
            macro_f1 = f1_score(y_ev, y_pred, average="macro", zero_division=0)
            present_classes, target_names = get_target_names(y_ev, y_pred, class_names)
            report = classification_report(
                y_ev, y_pred, labels=present_classes, target_names=target_names, output_dict=True, zero_division=0
            )

            results_by_split[sp][name] = {
                "accuracy": round(float(acc), 4),
                "train_accuracy": round(float(train_acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "train_time_sec": round(float(train_time), 3),
                "for_class": report
            }

            print(f"\n[{sp.upper()}] {name}")
            print(f"accuracy: {acc:.4f} | train acc: {train_acc:.4f} | F1: {macro_f1:.4f} | time: {train_time:.2f}s")
            print(classification_report(y_ev, y_pred, labels=present_classes, target_names=target_names, zero_division=0))

            save_confusion_matrix(
                y_ev,
                y_pred,
                labels=target_names,
                title=f"{name} ({args.feature_type} - {sp}) - normalized CM",
                save_path=out_dir / f"cm_{name.lower()}.png"
            )

    # train MyCNN once on raw 32x32x3 crops (gated on 'color' run to avoid duplicate training)
    if args.feature_type == "color":
        cnn_epochs = 2 if args.debug else 15
        cnn = MyCNN(epochs=cnn_epochs, batch_size=128, lr=1e-3, class_weight=cw)
        t0 = time.perf_counter()
        cnn.fit(x_train_crops, y_train)
        train_time = time.perf_counter() - t0
        train_acc = accuracy_score(y_train, cnn.predict(x_train_crops))

        # save checkpoint for T2 evaluation
        ckpt_path = base_out / "mycnn_best.pt"
        torch.save(
            {
                "state_dict": cnn.net.state_dict(),
                "classes": cnn.classes,
                "mean": cnn.mean,
                "std": cnn.std
            },
            ckpt_path
        )
        print(f"\nsaved MyCNN weights to {ckpt_path}")

        for sp, (_, y_ev, x_ev_crops) in eval_features.items():
            out_dir = base_out / sp / args.feature_type
            out_dir.mkdir(parents=True, exist_ok=True)

            y_pred = cnn.predict(x_ev_crops)
            acc = accuracy_score(y_ev, y_pred)
            macro_f1 = f1_score(y_ev, y_pred, average="macro", zero_division=0)
            present_classes, target_names = get_target_names(y_ev, y_pred, class_names)
            report = classification_report(
                y_ev, y_pred, labels=present_classes, target_names=target_names, output_dict=True, zero_division=0
            )

            results_by_split[sp]["MyCNN"] = {
                "accuracy": round(float(acc), 4),
                "train_accuracy": round(float(train_acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "train_time_sec": round(float(train_time), 3),
                "for_class": report
            }

            print(f"\n[{sp.upper()}] MyCNN")
            print(f"accuracy: {acc:.4f} | train acc: {train_acc:.4f} | F1: {macro_f1:.4f} | time: {train_time:.2f}s")
            print(classification_report(y_ev, y_pred, labels=present_classes, target_names=target_names, zero_division=0))

            save_confusion_matrix(
                y_ev,
                y_pred,
                labels=target_names,
                title=f"MyCNN (raw_crops - {sp}) - normalized CM",
                save_path=out_dir / "cm_mycnn.png"
            )

    # save JSON metrics separately for split
    for sp, sp_results in results_by_split.items():
        json_path = base_out / sp / args.feature_type / "metrics.json"
        print(f"Saving results in {json_path}...")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sp_results, f, indent=2)


if __name__ == "__main__":
    main()