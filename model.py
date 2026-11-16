import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
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


class BasePerceptron:
    def __init__(self, eta: float = 0.01, max_n: int = 20):
        self.eta = eta
        self.max_n = max_n
        self.w: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> Self:
        n, d = x.shape
        self.classes = np.unique(t)
        k = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # for each of the k classes target in {-1,1} -> T(N, K)
        T = -np.ones((n, k), dtype=np.float32)
        for i, label in enumerate(t):
            T[i, class_to_idx[label]] = 1.0

        phi = np.c_[np.ones(n, dtype=x.dtype), x]  # (N, D+1)
        rand_g = np.random.default_rng(42)
        self.w = rand_g.normal(0.0, 0.01, size=(d + 1, k)).astype(np.float32)

        for mn in range(self.max_n):
            # mixed the examples at each epoch
            errors=0
            indices = rand_g.permutation(n)
            for idx in indices:
                xk = phi[idx]
                tk = T[idx]
                # prediction {-1, +1} on all the 7 classes
                o = np.where(np.dot(xk, self.w) >= 0, 1.0, -1.0)
                if not np.array_equal(tk, o):
                    # update of weights
                    self.w += self.eta * np.outer(xk, tk - o)
                    errors+=1
            if errors==0:
                break

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.classes is None:
            raise RuntimeError("call fit() before predict().")
        phi = np.c_[np.ones(len(x), dtype=x.dtype), x]
        scores = np.matmul(phi, self.w)  # shape (N, K)
        return self.classes[np.argmax(scores, axis=1)]

def get_models( ls_lambda: float = 10.0, svm_c: float = 1.0,
    class_weight: str | None = "balanced") -> dict:
    return {
        "LeastSquare": LeastSquare(ls_lambda=ls_lambda),
        "BasePerceptron": BasePerceptron(eta=0.01, max_n=20),
        "LDA": LinearDiscriminantAnalysis(),
        "LinearSVC": LinearSVC(
            C=svm_c,
            class_weight=class_weight,
            max_iter=5000,
            random_state=42,
        )
    }