import numpy as np
from PIL import Image
from datasets import load_dataset


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    compute the Intersection over Union (IoU) element for element.
    boxes_a, boxes_b: array (N, 4) in format [x, y, w, h].
    """
    ax1, ay1 = boxes_a[:, 0], boxes_a[:, 1]
    ax2, ay2 = ax1 + np.maximum(boxes_a[:, 2], 0.0), ay1 + np.maximum(boxes_a[:, 3], 0.0)

    bx1, by1 = boxes_b[:, 0], boxes_b[:, 1]
    bx2, by2 = bx1 + np.maximum(boxes_b[:, 2], 0.0), by1 + np.maximum(boxes_b[:, 3], 0.0)

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return np.where(union > 1e-8, inter_area / union, 0.0).astype(np.float32)


def encode_boxes(gt_boxes: np.ndarray, crop_boxes: np.ndarray) -> np.ndarray:
    """
    compute the ground-truth box compared to the sampled windows:
    tx = (cx_gt - cx_c) / w_c
    ty = (cy_gt - cy_c) / h_c
    tw = log(w_gt / w_c)
    th = log(h_gt / h_c)
    """
    cx_gt = gt_boxes[:, 0] + 0.5 * gt_boxes[:, 2]
    cy_gt = gt_boxes[:, 1] + 0.5 * gt_boxes[:, 3]
    w_gt = np.maximum(gt_boxes[:, 2], 1e-4)
    h_gt = np.maximum(gt_boxes[:, 3], 1e-4)

    cx_c = crop_boxes[:, 0] + 0.5 * crop_boxes[:, 2]
    cy_c = crop_boxes[:, 1] + 0.5 * crop_boxes[:, 3]
    w_c = np.maximum(crop_boxes[:, 2], 1e-4)
    h_c = np.maximum(crop_boxes[:, 3], 1e-4)

    tx = (cx_gt - cx_c) / w_c
    ty = (cy_gt - cy_c) / h_c
    tw = np.log(w_gt / w_c)
    th = np.log(h_gt / h_c)

    return np.stack([tx, ty, tw, th], axis=1).astype(np.float32)


def decode_boxes(preds: np.ndarray, crop_boxes: np.ndarray) -> np.ndarray:
    """
    inverse transformation: reconstructs [x_pred, y_pred, w_pred, h_pred]
    in the original image based on the predicted offsets.
    """
    cx_c = crop_boxes[:, 0] + 0.5 * crop_boxes[:, 2]
    cy_c = crop_boxes[:, 1] + 0.5 * crop_boxes[:, 3]
    w_c = crop_boxes[:, 2]
    h_c = crop_boxes[:, 3]

    pred_cx = preds[:, 0] * w_c + cx_c
    pred_cy = preds[:, 1] * h_c + cy_c
    pred_w = w_c * np.exp(np.clip(preds[:, 2], -3.0, 3.0))
    pred_h = h_c * np.exp(np.clip(preds[:, 3], -3.0, 3.0))

    pred_x = pred_cx - 0.5 * pred_w
    pred_y = pred_cy - 0.5 * pred_h

    return np.stack([pred_x, pred_y, pred_w, pred_h], axis=1).astype(np.float32)


def _sample_jittered_box(
    gt_box: np.ndarray,
    img_w: int,
    img_h: int,
    rng: np.random.Generator,
    min_iou: float = 0.30
) -> np.ndarray:
    """
    sample a casual window around the ground-truth box ensuring
    a minimum overlap (min_iou) so that the object remains visible.
    """
    x, y, w, h = gt_box
    cx_gt = x + 0.5 * w
    cy_gt = y + 0.5 * h

    for attempt in range(20):
        # jitter more soft if the first attempts do not exceed min_iou
        scale_range = (0.75, 1.45) if attempt < 15 else (0.90, 1.15)
        shift_range = 0.25 if attempt < 15 else 0.08

        w_c = np.clip(w * rng.uniform(*scale_range), 4.0, float(img_w))
        h_c = np.clip(h * rng.uniform(*scale_range), 4.0, float(img_h))

        cx_c = cx_gt + rng.uniform(-shift_range, shift_range) * w
        cy_c = cy_gt + rng.uniform(-shift_range, shift_range) * h

        x1 = np.clip(cx_c - 0.5 * w_c, 0.0, max(0.0, img_w - 4.0))
        y1 = np.clip(cy_c - 0.5 * h_c, 0.0, max(0.0, img_h - 4.0))
        x2 = np.clip(x1 + w_c, x1 + 4.0, float(img_w))
        y2 = np.clip(y1 + h_c, y1 + 4.0, float(img_h))

        cand = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        iou = compute_iou(cand[None, :], gt_box[None, :])[0]
        if iou >= min_iou:
            return cand

    return gt_box.astype(np.float32)


def one_hot_classes(class_ids: np.ndarray, num_classes: int = 7) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[class_ids]


def load_regression_split(
    split: str = "train",
    patch_size: int = 32,
    samples_per_box: int = 1,
    debug: bool = False,
    seed: int = 42
):
    """
    load the HSLVision split and generate RGB patches (N, 32, 32, 3),
    relative targets (N, 4), crop_boxes (N, 4) and gt_boxes (N, 4)
    """
    ds = load_dataset("whirlwind-ams/hslvision", split=split)
    if debug:
        ds = ds.select(range(min(150, len(ds))))

    rng = np.random.default_rng(seed)
    patches, class_ids, crop_boxes, gt_boxes = [], [], [], []

    for row in ds:
        img: Image.Image = row["image"].convert("RGB")
        img_w, img_h = img.size
        bboxes = row["objects"]["bbox"]
        cats = row["objects"].get("category", row["objects"].get("label"))

        for raw_box, cls_id in zip(bboxes, cats):
            x, y, w, h = [float(v) for v in raw_box]
            x1 = np.clip(x, 0.0, img_w - 1.0)
            y1 = np.clip(y, 0.0, img_h - 1.0)
            x2 = np.clip(x + w, x1 + 1.0, float(img_w))
            y2 = np.clip(y + h, y1 + 1.0, float(img_h))
            gt_w, gt_h = x2 - x1, y2 - y1
            if gt_w < 4.0 or gt_h < 4.0:
                continue

            gt_box = np.array([x1, y1, gt_w, gt_h], dtype=np.float32)

            for _ in range(samples_per_box):
                c_box = _sample_jittered_box(gt_box, img_w, img_h, rng)
                cx1, cy1 = int(round(c_box[0])), int(round(c_box[1]))
                cx2 = max(cx1 + 1, min(img_w, int(round(c_box[0] + c_box[2]))))
                cy2 = max(cy1 + 1, min(img_h, int(round(c_box[1] + c_box[3]))))
                actual_crop = np.array(
                    [float(cx1), float(cy1), float(cx2 - cx1), float(cy2 - cy1)],
                    dtype=np.float32
                )

                patch = img.crop((cx1, cy1, cx2, cy2)).resize(
                    (patch_size, patch_size), Image.BILINEAR
                )
                patches.append(np.asarray(patch, dtype=np.float32) / 255.0)
                class_ids.append(int(cls_id))
                crop_boxes.append(actual_crop)
                gt_boxes.append(gt_box)

    patches_arr = np.stack(patches, axis=0)
    class_arr = np.array(class_ids, dtype=np.int64)
    crop_arr = np.stack(crop_boxes, axis=0)
    gt_arr = np.stack(gt_boxes, axis=0)
    targets_arr = encode_boxes(gt_arr, crop_arr)

    return patches_arr, class_arr, targets_arr, crop_arr, gt_arr