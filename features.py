import numpy as np
from skimage.feature import hog
from skimage.color import rgb2hsv, rgb2gray
from sklearn.preprocessing import StandardScaler


def extract_pixels(x_crops: np.ndarray) -> np.ndarray:
    """transform each crop (H, W, 3) into a row vector (H*W*3)."""
    return x_crops.reshape(len(x_crops), -1)


def extract_color_features(x_crops: np.ndarray, bins: int = 16) -> np.ndarray:
    """extracts color histograms and HOG descriptors for each crop"""
    features = []
    for img in x_crops:
        # histograms
        hsv = rgb2hsv(img)
        hu, _ = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1), density=True)
        sat, _ = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1), density=True)
        val, _ = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1), density=True)
        color_feat = np.concatenate([hu, sat, val])
        # descriptors
        gray = rgb2gray(img)
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                       feature_vector=True)
        features.append(np.concatenate([color_feat, hog_feat]))

    return np.array(features, dtype=np.float32)


def prepare_features(x_train: np.ndarray, x_val: np.ndarray, 
                     mode: str = "pixels") -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """converts the crops into standardized 2D matrices for the classifiers"""
    if mode == "pixels":
        x_tra_feat = extract_pixels(x_train)
        x_val_feat = extract_pixels(x_val)
    elif mode == "color":
        x_tra_feat = extract_color_features(x_train)
        x_val_feat = extract_color_features(x_val)
    else:
        raise ValueError(f"not supported: {mode}")

    # standardization
    scaler = StandardScaler()
    x_tra_scaled = scaler.fit_transform(x_tra_feat)
    x_val_scaled = scaler.transform(x_val_feat)

    return x_tra_scaled, x_val_scaled, scaler