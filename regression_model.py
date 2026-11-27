import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from regression_dataset import decode_boxes, compute_iou


class LeastSquareRegressor:
    """Linear regression multi-output with regularization Ridge L2"""

    def __init__(self, ls_lambda: float = 10.0):
        self.ls_lambda = ls_lambda
        self.W = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        N, D = X.shape
        Phi = np.hstack([np.ones((N, 1), dtype=np.float32), X])
        reg = np.eye(D + 1, dtype=np.float32) * self.ls_lambda
        reg[0, 0] = 0.0  # doesn't penalize the bias term
        self.W = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        return (Phi @ self.W).astype(np.float32)


class CNNRegNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 4 + num_classes, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

    def forward(self, x, cls_onehot):
        feat = torch.flatten(self.features(x), start_dim=1)
        combined = torch.cat([feat, cls_onehot], dim=1)
        return self.regressor(combined)


class MyCNNRegressor:
    """CNN end-to-end on RGB 32x32 patch trained with Smooth L1 (Huber) Loss"""

    def __init__(self, lr: float = 1e-3, epochs: int = 15, batch_size: int = 128, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = None
        self.std = None
        self.net = None

    def _preprocess(self, patches: np.ndarray, fit: bool = False) -> torch.Tensor:
        # (N, H, W, C) -> (N, C, H, W)
        x = np.transpose(patches, (0, 3, 1, 2)).astype(np.float32)
        if fit:
            self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
            self.std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        x = (x - self.mean) / self.std
        return torch.from_numpy(x)

    def fit(
        self,
        X_patches: np.ndarray,
        X_cls: np.ndarray,
        Y: np.ndarray,
        X_val_patches: np.ndarray = None,
        X_val_cls: np.ndarray = None,
        val_crop_boxes: np.ndarray = None,
        val_gt_boxes: np.ndarray = None
    ):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.net = CNNRegNet(num_classes=X_cls.shape[1]).to(self.device)
        X_t = self._preprocess(X_patches, fit=True)
        C_t = torch.from_numpy(X_cls.astype(np.float32))
        Y_t = torch.from_numpy(Y.astype(np.float32))

        loader = DataLoader(TensorDataset(X_t, C_t, Y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()

        best_iou = -1.0
        best_weights = copy.deepcopy(self.net.state_dict())

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running_loss = 0.0
            for bx, bc, by in loader:
                bx, bc, by = bx.to(self.device), bc.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.net(bx, bc)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * bx.size(0)

            train_loss = running_loss / len(X_t)

            if X_val_patches is not None and X_val_cls is not None and val_crop_boxes is not None:
                val_preds = self.predict(X_val_patches, X_val_cls)
                pred_boxes = decode_boxes(val_preds, val_crop_boxes)
                val_iou = float(compute_iou(pred_boxes, val_gt_boxes).mean())
                print(f"  [MyCNNReg] Epoch {epoch:02d}/{self.epochs} | loss={train_loss:.4f} | val_mIoU={val_iou:.4f}")
                if val_iou > best_iou:
                    best_iou = val_iou
                    best_weights = copy.deepcopy(self.net.state_dict())
            else:
                print(f"  [MyCNNReg] Epoch {epoch:02d}/{self.epochs} | loss={train_loss:.4f}")

        if X_val_patches is not None and X_val_cls is not None:
            self.net.load_state_dict(best_weights)
        return self

    def predict(self, X_patches: np.ndarray, X_cls: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = self._preprocess(X_patches, fit=False)
        C_t = torch.from_numpy(X_cls.astype(np.float32))
        loader = DataLoader(TensorDataset(X_t, C_t), batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, bc in loader:
                bx, bc = bx.to(self.device), bc.to(self.device)
                preds.append(self.net(bx, bc).cpu().numpy())
        return np.vstack(preds).astype(np.float32)


def get_regression_models(epochs: int = 15):
    return {
        "LeastSquareReg": LeastSquareRegressor(ls_lambda=10.0),
        "KNNReg": KNeighborsRegressor(n_neighbors=11, weights="distance", n_jobs=-1),
        "BoostingReg": MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
        ),
        "MyCNNReg": MyCNNRegressor(lr=1e-3, epochs=epochs, batch_size=128)
    }