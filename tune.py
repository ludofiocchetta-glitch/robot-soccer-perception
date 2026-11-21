import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from dataset import load_split
from features import prepare_features
from model import LeastSquare, MyCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="hyperparameter search on validation split")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="fast sanity check on a small subset of samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tuning",
        help="directory where JSON metrics and plots will be saved"
    )
    return parser.parse_args()


def evaluate_candidate(
    clf,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    y_train_pred = clf.predict(x_train)
    y_val_pred = clf.predict(x_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    return {
        "train_accuracy": round(float(train_acc), 4),
        "val_accuracy": round(float(val_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "train_time_sec": round(float(train_time), 3)
    }


def save_tuning_plot(all_results: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    axes = axes.flatten()

    configs = [
        ("LeastSquare", "ls_lambda", "LeastSquare - L2 Regularization (lambda)"),
        ("KNN", "n_neighbors", "KNN - Number of Neighbors (k)"),
        ("Boosting", "learning_rate", "Boosting - Learning Rate"),
        ("MyCNN", "lr", "MyCNN - Adam Learning Rate")
    ]

    for ax, (model_name, param_name, title) in zip(axes, configs):
        entries = all_results[model_name]
        x_labels = [str(item[param_name]) for item in entries]
        x_pos = np.arange(len(x_labels))

        val_accs = [item["val_accuracy"] for item in entries]
        macro_f1s = [item["macro_f1"] for item in entries]

        ax.plot(x_pos, val_accs, marker="o", linewidth=2, label="accuracy")
        ax.plot(x_pos, macro_f1s, marker="s", linestyle="--", linewidth=2, label="macro F1")

        best_idx = int(np.argmax(macro_f1s))
        ax.scatter(
            [x_pos[best_idx]],
            [macro_f1s[best_idx]],
            color="blue",
            zorder=5,
            s=70,
            label=f"best F1 ({x_labels[best_idx]}: {macro_f1s[best_idx]:.4f})"
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_train = 150 if args.debug else None
    max_val = 50 if args.debug else None

    print(f"loading train and valid splits (debug={args.debug})...")
    x_train_crops, y_train = load_split(split_name="train", max_samples=max_train, streaming=args.debug)
    x_val_crops, y_val = load_split(split_name="valid", max_samples=max_val, streaming=args.debug)
    print(f"train crops: {len(y_train)} | valid crops: {len(y_val)}")

    print("extracting standardized 'color' features...")
    x_train, x_val, _ = prepare_features(x_train_crops, x_val_crops, mode="color")

    all_results: dict[str, list[dict]] = {
        "LeastSquare": [],
        "KNN": [],
        "Boosting": [],
        "MyCNN": []
    }

    # leastSquare: L2 regularization lambda
    print("\n--- Tuning LeastSquare (ls_lambda) ---")
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        clf = LeastSquare(ls_lambda=lam)
        metrics = evaluate_candidate(clf, x_train, y_train, x_val, y_val)
        metrics["ls_lambda"] = lam
        all_results["LeastSquare"].append(metrics)
        print(f"ls_lambda={lam:<5} | val_acc={metrics['val_accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | time={metrics['train_time_sec']:.2f}s")

    # KNN: number of neighbors k
    print("\n--- Tuning KNN (n_neighbors) ---")
    k_grid = [1, 3, 5, 11, 21] if not args.debug else [1, 3, 5]
    for k in k_grid:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean", n_jobs=-1)
        metrics = evaluate_candidate(clf, x_train, y_train, x_val, y_val)
        metrics["n_neighbors"] = k
        all_results["KNN"].append(metrics)
        print(f"n_neighbors={k:<2}  | val_acc={metrics['val_accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | time={metrics['train_time_sec']:.2f}s")

    # boosting: learning_rate
    print("\n--- Tuning Boosting (learning_rate) ---")
    lr_boost_grid = [0.02, 0.05, 0.1, 0.2] if not args.debug else [0.05, 0.1]
    for lr_b in lr_boost_grid:
        clf = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=200,
            learning_rate=lr_b,
            random_state=42
        )
        metrics = evaluate_candidate(clf, x_train, y_train, x_val, y_val)
        metrics["learning_rate"] = lr_b
        all_results["Boosting"].append(metrics)
        print(f"learning_rate={lr_b:<4} | val_acc={metrics['val_accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | time={metrics['train_time_sec']:.2f}s")

    # myCNN: learning rate on raw 32x32x3 crops
    print("\n--- Tuning MyCNN (lr) ---")
    cnn_epochs = 2 if args.debug else 15
    lr_cnn_grid = [1e-4, 5e-4, 1e-3, 5e-3] if not args.debug else [1e-4, 1e-3]
    for lr_c in lr_cnn_grid:
        clf = MyCNN(epochs=cnn_epochs, batch_size=128, lr=lr_c, class_weight="balanced")
        metrics = evaluate_candidate(clf, x_train_crops, y_train, x_val_crops, y_val)
        metrics["lr"] = lr_c
        all_results["MyCNN"].append(metrics)
        print(f"lr={lr_c:<6} | val_acc={metrics['val_accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | time={metrics['train_time_sec']:.2f}s")

    json_path = out_dir / "tuning_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    plot_path = out_dir / "hyperparameter_search.png"
    save_tuning_plot(all_results, plot_path)

    print(f"\nsaved JSON results to: {json_path}")
    print(f"saved 2x2 tuning plot to: {plot_path}")


if __name__ == "__main__":
    main()