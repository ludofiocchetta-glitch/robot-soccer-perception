import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from features import prepare_features
from regression_dataset import (
    compute_iou,
    decode_boxes,
    load_regression_split,
    one_hot_classes
)
from regression_model import LeastSquareRegressor, MyCNNRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression hyperparameter search on validation split")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="fast sanity check on a small subset of images"
    )
    parser.add_argument(
        "--samples_per_box",
        type=int,
        default=2,
        help="jittered crops sampled per ground-truth box"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/tuning",
        help="directory where JSON metrics and plots will be saved"
    )
    return parser.parse_args()


def evaluate_candidate(
    model,
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    val_crop_boxes: np.ndarray,
    val_gt_boxes: np.ndarray,
    is_cnn: bool = False,
    c_train: np.ndarray = None,
    c_val: np.ndarray = None
) -> dict:
    t0 = time.perf_counter()
    if is_cnn:
        model.fit(
            x_train,
            c_train,
            y_train,
            X_val_patches=x_val,
            X_val_cls=c_val,
            val_crop_boxes=val_crop_boxes,
            val_gt_boxes=val_gt_boxes
        )
    else:
        model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    if is_cnn:
        y_val_pred = model.predict(x_val, c_val)
    else:
        y_val_pred = model.predict(x_val)

    pred_boxes = decode_boxes(y_val_pred, val_crop_boxes)
    val_miou = float(compute_iou(pred_boxes, val_gt_boxes).mean())
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    return {
        "val_mIoU": round(val_miou, 4),
        "val_mse": round(val_mse, 4),
        "val_mae": round(val_mae, 4),
        "train_time_sec": round(float(train_time), 3)
    }


def save_tuning_plot(all_results: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    axes = axes.flatten()

    configs = [
        ("LeastSquareReg", "ls_lambda", "LeastSquareReg - L2 Regularization (lambda)"),
        ("KNNReg", "n_neighbors", "KNNReg - Number of Neighbors (k)"),
        ("BoostingReg", "learning_rate", "BoostingReg - Learning Rate"),
        ("MyCNNReg", "lr", "MyCNNReg - Adam Learning Rate")
    ]

    for ax, (model_name, param_name, title) in zip(axes, configs):
        entries = all_results[model_name]
        x_labels = [str(item[param_name]) for item in entries]
        x_pos = np.arange(len(x_labels))

        val_mious = [item["val_mIoU"] for item in entries]
        val_maes = [item["val_mae"] for item in entries]

        line1 = ax.plot(x_pos, val_mious, color="tab:blue", marker="o", linewidth=2, label="val mIoU (higher better)")

        best_idx = int(np.argmax(val_mious))
        best_pt = ax.scatter(
            [x_pos[best_idx]],
            [val_mious[best_idx]],
            color="navy",
            zorder=5,
            s=75,
            label=f"best mIoU ({x_labels[best_idx]}: {val_mious[best_idx]:.4f})"
        )

        ax2 = ax.twinx()
        line2 = ax2.plot(
            x_pos,
            val_maes,
            color="tab:orange",
            marker="s",
            linestyle="--",
            linewidth=2,
            label="val MAE (lower better)"
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Validation mIoU", color="tab:blue")
        ax2.set_ylabel("Validation MAE", color="tab:orange")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)

        handles = line1 + [best_pt] + line2
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_train = 150 if args.debug else None
    max_val = 50 if args.debug else None

    print(f"loading train and valid regression splits (debug={args.debug})...")
    train_patches, train_cls, y_train, _, _ = load_regression_split(
        split="train",
        samples_per_box=args.samples_per_box,
        max_images=max_train,
        patch_size=32
    )
    val_patches, val_cls, y_val, val_crop_boxes, val_gt_boxes = load_regression_split(
        split="valid",
        samples_per_box=1,
        max_images=max_val,
        patch_size=32
    )
    print(f"train samples: {len(y_train)} | valid samples: {len(y_val)}")

    num_classes = int(max(train_cls.max(), val_cls.max())) + 1
    c_train = one_hot_classes(train_cls, num_classes=num_classes)
    c_val = one_hot_classes(val_cls, num_classes=num_classes)

    print("extracting standardized 'color' features + class one-hot...")
    x_train_color, x_val_color, _ = prepare_features(train_patches, val_patches, mode="color")
    x_train_tab = np.hstack([x_train_color, c_train])
    x_val_tab = np.hstack([x_val_color, c_val])

    all_results: dict[str, list[dict]] = {
        "LeastSquareReg": [],
        "KNNReg": [],
        "BoostingReg": [],
        "MyCNNReg": []
    }

    # 1. LeastSquareReg: L2 regularization lambda
    print("\n--- Tuning LeastSquareReg (ls_lambda) ---")
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        model = LeastSquareRegressor(ls_lambda=lam)
        metrics = evaluate_candidate(model, x_train_tab, y_train, x_val_tab, y_val, val_crop_boxes, val_gt_boxes)
        metrics["ls_lambda"] = lam
        all_results["LeastSquareReg"].append(metrics)
        print(
            f"ls_lambda={lam:<5} | val_mIoU={metrics['val_mIoU']:.4f} | "
            f"val_MSE={metrics['val_mse']:.4f} | val_MAE={metrics['val_mae']:.4f} | time={metrics['train_time_sec']:.2f}s"
        )

    # 2. KNNReg: number of neighbors k
    print("\n--- Tuning KNNReg (n_neighbors) ---")
    k_grid = [3, 5, 7, 11, 15] if not args.debug else [1, 3, 5]
    for k in k_grid:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="euclidean", n_jobs=-1)
        metrics = evaluate_candidate(model, x_train_tab, y_train, x_val_tab, y_val, val_crop_boxes, val_gt_boxes)
        metrics["n_neighbors"] = k
        all_results["KNNReg"].append(metrics)
        print(
            f"n_neighbors={k:<2}  | val_mIoU={metrics['val_mIoU']:.4f} | "
            f"val_MSE={metrics['val_mse']:.4f} | val_MAE={metrics['val_mae']:.4f} | time={metrics['train_time_sec']:.2f}s"
        )

    # 3. BoostingReg: learning_rate
    print("\n--- Tuning BoostingReg (learning_rate) ---")
    lr_boost_grid = [0.03, 0.05, 0.1, 0.2] if not args.debug else [0.05, 0.1]
    for lr_b in lr_boost_grid:
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, learning_rate=lr_b, random_state=42)
        )
        metrics = evaluate_candidate(model, x_train_tab, y_train, x_val_tab, y_val, val_crop_boxes, val_gt_boxes)
        metrics["learning_rate"] = lr_b
        all_results["BoostingReg"].append(metrics)
        print(
            f"learning_rate={lr_b:<4} | val_mIoU={metrics['val_mIoU']:.4f} | "
            f"val_MSE={metrics['val_mse']:.4f} | val_MAE={metrics['val_mae']:.4f} | time={metrics['train_time_sec']:.2f}s"
        )

    # 4. MyCNNReg: Adam learning rate on raw 32x32x3 patches + class one-hot
    print("\n--- Tuning MyCNNReg (lr) ---")
    cnn_epochs = 2 if args.debug else 15
    lr_cnn_grid = [1e-4, 5e-4, 1e-3, 3e-3] if not args.debug else [1e-4, 1e-3]
    for lr_c in lr_cnn_grid:
        model = MyCNNRegressor(lr=lr_c, epochs=cnn_epochs, batch_size=128)
        metrics = evaluate_candidate(
            model,
            train_patches,
            y_train,
            val_patches,
            y_val,
            val_crop_boxes,
            val_gt_boxes,
            is_cnn=True,
            c_train=c_train,
            c_val=c_val
        )
        metrics["lr"] = lr_c
        all_results["MyCNNReg"].append(metrics)
        print(
            f"lr={lr_c:<6} | val_mIoU={metrics['val_mIoU']:.4f} | "
            f"val_MSE={metrics['val_mse']:.4f} | val_MAE={metrics['val_mae']:.4f} | time={metrics['train_time_sec']:.2f}s"
        )

    json_path = out_dir / "regression_tuning_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    plot_path = out_dir / "regression_hyperparameter_search.png"
    save_tuning_plot(all_results, plot_path)

    print(f"\nsaved JSON results to: {json_path}")
    print(f"saved 2x2 tuning plot to: {plot_path}")


if __name__ == "__main__":
    main()