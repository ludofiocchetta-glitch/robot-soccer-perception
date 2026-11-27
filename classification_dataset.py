from datasets import load_dataset, load_dataset_builder
import numpy as np
from PIL import Image

def load_split(
    split_name: str = "train", 
    max_samples: int | None = None,
    img_size: tuple[int, int] = (32, 32),
    margin_ratio: float = 0.05,
    streaming: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    load a split and return (x, y) for classification
    """
    ds = load_dataset("whirlwind-ams/hslvision", split=split_name, streaming=streaming)

    # check the dimension
    if max_samples is not None:
        ds = ds.take(max_samples)

    x_crops = []
    y_labels = []
    for sample in ds:
        image: Image.Image = sample["image"].convert("RGB")
        dim_w= sample["width"]
        dim_h = sample["height"]
        objects = sample["objects"]

        for bbox, category in zip(objects["bbox"], objects["category"]):
            x, y, w, h = bbox

            # add a small margin
            pad_w = w * margin_ratio
            pad_h = h * margin_ratio

            x1 = max(0, int(round(x - pad_w)))
            y1 = max(0, int(round(y - pad_h)))
            x2 = min(dim_w, int(round(x + w + pad_w)))
            y2 = min(dim_h, int(round(y + h + pad_h)))
            # discard the boxes out of the frame
            if x2 <= x1 or y2 <= y1:
                    continue
            
            # crop, resize and normalization 
            crop = image.crop((x1, y1, x2, y2))
            crop_resized = crop.resize(img_size, Image.Resampling.BILINEAR)
            crop_array = np.asarray(crop_resized, dtype=np.float32) / 255.0

            x_crops.append(crop_array)
            y_labels.append(category)

    return np.array(x_crops), np.array(y_labels)

def get_class_names() -> list[str]:
    builder = load_dataset_builder("whirlwind-ams/hslvision")
    cat_feat = builder.info.features["objects"]["category"]
    if hasattr(cat_feat, "feature") and hasattr(cat_feat.feature, "names"):
        return cat_feat.feature.names
    if hasattr(cat_feat, "names"):
        return cat_feat.names
    return []


if __name__ == "__main__":
    x_test, y_test = load_split(split_name="train", max_samples=20)
    print(f"extracted objects: {len(x_test)}")
    print(f"shape of x: {x_test.shape}, shape of y: {y_test.shape}")
    print(f"number of examples for each class: {np.unique(y_test, return_counts=True)}")
