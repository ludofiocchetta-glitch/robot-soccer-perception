import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from features import extract_features
from regression_dataset import (
    load_regression_split,
    one_hot_classes,
    decode_boxes,
    compute_iou
)
from regression_model import get_regression_models


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    crop_boxes: np.ndarray,
    gt_boxes: np.ndarray
):
    pred_boxes = decode_boxes(preds, crop_boxes)
    ious = compute_iou(pred_boxes, gt_boxes)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))

    return {
        "iou": float(np.mean(ious)),
        "mse": mse,
        "mae": mae
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--samples_per_box", type=int, default=2, help="Crops sampled per box on train split")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results_regression")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("1. Loading and generating jittered patches (train, valid, test)...")
    train_patches, train_cls, Y_train, train_crops, train_gts = load_regression_split(
        "train", samples_per_box=args.samples_per_box, debug=args.debug, seed=42
    )
    val_patches, val_cls, Y_val, val_crops, val_gts = load_regression_split(
        "valid", samples_per_box=1, debug=args.debug, seed=100
    )
    test_patches, test_cls, Y_test, test_crops, test_gts = load_regression_split(
        "test", samples_per_box=1, debug=args.debug, seed=200
    )
    print(f"Samples -> train: {len(train_patches)} | valid: {len(val_patches)} | test: {len(test_patches)}")

    # Baseline before the regression
    init_val = evaluate_predictions(np.zeros_like(Y_val), Y_val, val_crops, val_gts)
    init_test = evaluate_predictions(np.zeros_like(Y_test), Y_test, test_crops, test_gts)
    print(
        f"Baseline (No Regression) -> "
        f"VALID IoU={init_val['iou']:.4f} | MSE={init_val['mse']:.4f} | MAE={init_val['mae']:.4f} || "
        f"TEST IoU={init_test['iou']:.4f} | MSE={init_test['mse']:.4f} | MAE={init_test['mae']:.4f}"
    )

    print("2. Extracting 'color' features + class one-hot...")
    C_train = one_hot_classes(train_cls, num_classes=7)
    C_val = one_hot_classes(val_cls, num_classes=7)
    C_test = one_hot_classes(test_cls, num_classes=7)

    scaler = StandardScaler()
    X_train_color = scaler.fit_transform(extract_features(train_patches, feature_type="color"))
    X_val_color = scaler.transform(extract_features(val_patches, feature_type="color"))
    X_test_color = scaler.transform(extract_features(test_patches, feature_type="color"))

    X_train = np.hstack([X_train_color, C_train]).astype(np.float32)
    X_val = np.hstack([X_val_color, C_val]).astype(np.float32)
    X_test = np.hstack([X_test_color, C_test]).astype(np.float32)

    models = get_regression_models(epochs=args.epochs)
    summary = {
        "initial_window": {
            "valid": init_val,
            "test": init_test
        }
    }

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        if name == "MyCNNReg":
            model.fit(train_patches, C_train, Y_train, val_patches, C_val, val_crops, val_gts)
            val_preds = model.predict(val_patches, C_val)
            test_preds = model.predict(test_patches, C_test)
            torch.save(model.net.state_dict(), os.path.join(args.out_dir, "mycnn_reg_best.pt"))
        else:
            model.fit(X_train, Y_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)
        elapsed = time.time() - t0

        val_metrics = evaluate_predictions(val_preds, Y_val, val_crops, val_gts)
        test_metrics = evaluate_predictions(test_preds, Y_test, test_crops, test_gts)
        summary[name] = {
            "train_time_s": round(elapsed, 2),
            "valid": val_metrics,
            "test": test_metrics
        }

        print(
            f"[{name}] time={elapsed:.1f}s | "
            f"VALID IoU={val_metrics['iou']:.4f}, MSE={val_metrics['mse']:.4f}, MAE={val_metrics['mae']:.4f} | "
            f"TEST IoU={test_metrics['iou']:.4f}, MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}"
        )

    with open(os.path.join(args.out_dir, "regression_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # comparative graphic (IoU and MSE/MAE on test set)
    labels = list(summary.keys())
    test_ious = [summary[k]["test"]["iou"] for k in labels]
    test_mses = [summary[k]["test"]["mse"] for k in labels]
    test_maes = [summary[k]["test"]["mae"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_pos = np.arange(len(labels))

    axes[0].bar(x_pos, test_ious, width=0.5, color="#3f8bc1")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Mean IoU")
    axes[0].set_title("Intersection over Union")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1].bar(x_pos - 0.18, test_mses, width=0.35, label="MSE", color="#ff7f0e")
    axes[1].bar(x_pos + 0.18, test_maes, width=0.35, label="MAE", color="#202394")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("Error")
    axes[1].set_title("MSE & MAE")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "regression_comparison.png"), dpi=200)
    print(f"\nsaved metrics, weights and plot to '{args.out_dir}/'")


if __name__ == "__main__":
    main()