import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Self


class LeastSquare:
    def __init__(self, ls_lambda: float = 0.0):
        self.ls_lambda = ls_lambda
        self.w: np.ndarray | None = None
        self.classes : np.ndarray | None = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> Self:
        n, d = x.shape
        self.classes = np.unique(t)
        k = len(self.classes)
        # associate to each class the real index
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 1-of-k coding scheme for t -> T matrix 
        T = np.zeros((n, k), dtype=np.float32)
        for i, label in enumerate(t):
            T[i, class_to_idx[label]] = 1.0
        # design matrix (N, D+1) for hide w0
        phi = np.c_[np.ones(n, dtype=x.dtype), x]

        if self.ls_lambda > 0:
            # regolarization version for avoid overfitting 
            reg = self.ls_lambda * np.eye(d + 1, dtype=x.dtype)
            reg[0, 0] = 0.0  # not penalize the bias
            self.w = np.linalg.solve(phi.T @ phi + reg, phi.T @ T)
        else:
            # exact LS solution with pseudoinverse
            self.w = np.matmul(np.linalg.pinv(phi), T)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.classes is None:
            raise RuntimeError("call fit() before predict()")
        phi = np.c_[np.ones(len(x), dtype=x.dtype), x]
        scores = np.matmul(phi, self.w)  # shape (N, K)
        # selection of the winning class
        return self.classes[np.argmax(scores, axis=1)]


def get_models( ls_lambda: float = 10.0, svm_c: float = 1.0,
    class_weight: str | None = "balanced") -> dict:
    return {
        "LeastSquare": LeastSquare(ls_lambda=ls_lambda),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        ),
        "Boosting": HistGradientBoostingClassifier(
            class_weight=class_weight,
            max_iter=200,
            random_state=42
        )
    }

class MyCNN:
    """Wrapper scikit-learn style for CNN PyTorch on crops (N, 32, 32, 3)."""

    def __init__(
        self,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        class_weight: str | None = "balanced"
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: nn.Module | None = None
        self.classes: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _build_net(self, num_classes: int) -> nn.Module:
        return nn.Sequential(
            # input: (B, 3, 32, 32)
            # filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # normalization
            nn.BatchNorm2d(32),
            # non linearity for the convolution
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 64, 8, 8)
            # for complex patterns 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # preserves spatial geometry
            nn.MaxPool2d(2),  # -> (B, 128, 4, 4)
            # delete final ones
            nn.Flatten(),
            # turn off a random 30% of the 128 neurals
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # final linear layer 
            nn.Linear(128, num_classes)
        )

    def _preprocess(self, x_crops: np.ndarray, fit: bool = False) -> torch.Tensor:
        # from (N, H, W, C) to (N, C, H, W)
        x = np.transpose(x_crops, (0, 3, 1, 2)).astype(np.float32)
        if fit:
            self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
            self.std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        x = (x - self.mean) / self.std
        return torch.from_numpy(x)

    def fit(self, x_crops: np.ndarray, t: np.ndarray) -> Self:
        torch.manual_seed(42)
        self.classes = np.unique(t)
        k = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # labels to integers
        y_idx = np.array([class_to_idx[label] for label in t], dtype=np.int64)

        x_tensor = self._preprocess(x_crops, fit=True)
        y_tensor = torch.from_numpy(y_idx)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.net = self._build_net(k).to(self.device)

        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=k).astype(np.float32)
            weights = len(y_idx) / (k * np.maximum(counts, 1.0))
            # how much the prediction is far away from the real class
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        # update the weights
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(bx), by)
                # how every weight contributes to the error (criterion)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, x_crops: np.ndarray) -> np.ndarray:
        if self.net is None or self.classes is None:
            raise RuntimeError("Call fit() before predict().")
        x_tensor = self._preprocess(x_crops, fit=False)
        loader = DataLoader(TensorDataset(x_tensor), batch_size=self.batch_size, shuffle=False)
        # not more training
        self.net.eval()
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.net(bx)
                # predicts the index of the column with the max value
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        pred_idx = np.concatenate(preds, axis=0)
        # reconverts to the main class
        return self.classes[pred_idx]