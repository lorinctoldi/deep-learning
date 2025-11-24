import os
import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from collections import deque
from typing import Iterator, Sequence, List, Tuple, Dict, Any
from PIL import Image
from IPython.display import display
import cv2

from constants import SHAPE, CACHE_PATH, IMAGE_PATH


def load_masks(filter_empty: bool = False, original=False) -> pd.DataFrame:
    """
    Load the segmentation RLEs from CSV.

    :param filter_empty: If True, remove rows where 'EncodedPixels' is empty
    :param original: If True, load from original_segmentations.csv
    :return: DataFrame with columns 'ImageId' and 'EncodedPixels'
    """
    path = (
        "./data/original_segmentations.csv" if original else "./data/segmentations.csv"
    )
    masks = pd.read_csv(path).fillna("")
    if filter_empty:
        masks = masks[masks.EncodedPixels != ""]
    return masks


def get_mask_from_rle(rle: str | float | None, shape=SHAPE) -> np.ndarray:
    """
    Convert an RLE string to a 2D binary mask.

    :param rle_string: Run-length encoded mask as a string
    :param shape: Tuple of (height, width) for the output mask
    :return: 2D numpy array of 0s and 1s representing the mask
    """
    H, W = shape

    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((H, W), dtype=np.uint8)

    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros((H, W), dtype=np.uint8)
    
    mask = np.zeros(H * W, dtype=np.uint8)
    if rle.strip() == "":
        return mask.reshape((H, W))
    s = list(map(int, rle.split()))
    for start, length in zip(s[0::2], s[1::2]):
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape((H, W), order="F")

def get_rle_from_mask(mask: np.ndarray) -> str:
    """
    Convert a 2D binary mask to a run-length encoded (RLE) string.

    :param mask: 2D numpy array of 0s and 1s
    :return: RLE string (empty string if mask has no positive pixels)
    """
    pixels = mask.flatten(order="F")

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    starts = runs[::2]
    ends = runs[1::2]
    lengths = ends - starts

    rle = " ".join([f"{s} {l}" for s, l in zip(starts, lengths)])
    return rle


def rles_to_combined_mask(rle_list, shape=SHAPE):
    """
    Convert a list of RLE strings into a single 2D mask array.

    :param rle_list: List of RLE strings (one per boat)
    :param shape: Tuple of (height, width)
    :return: 2D numpy array with summed mask values; pixels >1 indicate overlap
    """
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        mask = get_mask_from_rle(rle, shape)
        combined_mask += mask
    return combined_mask


def has_overlap(mask) -> bool:
    """
    Check whether a 2D mask contains overlapping shapes.

    In a combined mask, overlapping areas (where multiple shapes occupy
    the same pixels) will have values greater than 1.

    :param mask: 2D numpy array representing a combined mask
    :return: True if any pixel has a value >1, indicating overlap; False otherwise
    """
    return np.any(mask > 1)  # type: ignore


def use_cache(func):
    def wrapper(*args, **kwargs):
        cache_file = os.path.join(CACHE_PATH, f"{func.__name__}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    return wrapper

def visualize_image_with_mask(image_id: str, mask: np.ndarray) -> None:
    """
    Visualize an image with its mask overlaid.

    :param image_id: File name of the image
    :param mask: 2D numpy array of 0s and 1s representing the mask
    """
    img = Image.open(f"./data/images/{image_id}").convert("RGBA")

    if mask.shape != (img.height, img.width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {(img.height, img.width)}"
        )

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    red_overlay = Image.new("RGBA", img.size, color=(255, 0, 0, 100))
    mask_overlay = Image.composite(red_overlay, Image.new("RGBA", img.size), mask_img)

    combined = Image.alpha_composite(img, mask_overlay)

    display(combined)

def get_number_of_shapes_and_sizes(mask: np.ndarray) -> list[int]:
    """
    Compute the sizes (number of pixels) of all connected shapes in a 2D binary grid.

    :param grid: A 2D grid of 0s and 1s representing a binary mask
    :return: List[int]: A list containing the size (number of pixels) of each shape
    found in the grid
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    sizes = stats[1:, cv2.CC_STAT_AREA].tolist()
    return sizes

def keep_largest_shape(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected shape in a 2D binary mask.
    All smaller shapes are removed.

    :param mask: 2D numpy array of 0s and 1s
    :return: new 2D mask with only the largest shape
    """
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)

    return largest_mask

def get_image(img_id: str) -> np.ndarray:
    """
    Load an image and its ground truth masks.

    :param img_id: Image filename
    :returns: image array
    """
    img = cv2.imread(f"{IMAGE_PATH}/{img_id}")

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}/{img_id}")

    return img[:, :, ::-1]  # BGR→RGB


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    :param a: First binary mask (2D numpy array of 0/1 or bool)
    :param b: Second binary mask (2D numpy array of 0/1 or bool)
    :return: IoU value as a float between 0.0 and 1.0
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def non_max_mask_suppression(masks, iou_thresh=0.8, score_key="predicted_iou"):
    """
    Apply non-maximum suppression to masks based on IoU and a score key.
    This removes overlapping masks, keeping the highest-scoring ones.

    :param masks: List of mask dicts from SAM
    :param iou_thresh: IoU threshold above which to suppress masks
    :param score_key: Key in mask dicts to use for scoring
    """
    masks_sorted = sorted(masks, key=lambda m: m[score_key], reverse=True)
    keep = []

    while masks_sorted:
        current = masks_sorted.pop(0)
        keep.append(current)

        masks_sorted = [
            m for m in masks_sorted
            if iou(current["segmentation"], m["segmentation"]) <= iou_thresh
        ]

    return keep


def enforce_no_overlap(
    masks: list[dict], score_key: str = "predicted_iou"
) -> list[dict]:
    """
    Remove pixel-level overlaps between masks by giving pixels to the higher-scoring mask.

    :param masks: List of mask dicts with 'segmentation' keys
    :param score_key: Key in mask dicts used to sort by score
    :return: List of masks with no overlapping pixels
    """
    masks_sorted = sorted(masks, key=lambda m: m[score_key], reverse=True)

    occupied = np.zeros_like(masks_sorted[0]["segmentation"], dtype=bool)
    final_masks = []

    for mask in masks_sorted:
        seg = mask["segmentation"].copy()
        seg[occupied] = False
        if seg.any():
            final_masks.append({**mask, "segmentation": seg})
            occupied |= seg  # mark these pixels as taken

    return final_masks


def compute_iou_matrix(g: List[np.ndarray], p: List[np.ndarray]) -> np.ndarray:
    """
    Compute IoU matrix between ground truth masks and predicted masks.

    :param g: List of ground truth masks (2D numpy arrays)
    :param p: List of predicted masks (2D numpy arrays)
    :returns: IoU matrix of shape (len(g), len(p))
    """
    iou_matrix = np.zeros((len(g), len(p)), dtype=np.float32)
    for i, gm in enumerate(g):
        for j, pm in enumerate(p):
            iou_matrix[i, j] = iou(gm, pm)
    return iou_matrix

def compute_confusion_counts(iou_mat: np.ndarray, t: np.float32) -> tuple[int, int, int]:
    """
    Compute true positives (TP), false negatives (FN), and false positives (FP) 
    from an IoU matrix at a given threshold.

    :param iou_mat: IoU matrix of shape (num_gt, num_pred)
    :param t: IoU threshold to consider a prediction as a match
    :return: Tuple (TP, FN, FP)
    """
    match_matrix = (iou_mat >= t)
    
    gt_matched = match_matrix.any(axis=1)
    pred_matched = match_matrix.any(axis=0)

    TP = gt_matched.sum()
    FN = (~gt_matched).sum()
    FP = (~pred_matched).sum()

    return int(TP), int(FN), int(FP)

def compute_f_score(t: np.float32, iou_mat: np.ndarray, beta: float = 2.0) -> float:
    """
    Compute the F-score at a given IoU threshold.

    :param t: IoU threshold
    :param iou_mat: IoU matrix of shape (num_gt, num_pred)
    :param beta: Weight of recall relative to precision (default 2.0)
    :return: F-score at threshold t
    """
    TP, FN, FP = compute_confusion_counts(iou_mat, t)

    num = (1 + beta**2) * TP
    den = (1 + beta**2) * TP + (beta**2) * FN + FP

    return num / den if den > 0 else 0.0

def compute_average_f_score(iou_mat: np.ndarray, beta: float = 2.0) -> np.float32:
    """
    Compute the average F-score across multiple IoU thresholds (0.5 to 0.95, step 0.05).

    :param iou_mat: IoU matrix of shape (num_gt, num_pred)
    :param beta: Weight of recall relative to precision (default 2.0)
    :return: Average F-score across thresholds
    """
    thresholds: NDArray[np.float32] = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    f_scores = [compute_f_score(t, iou_mat, beta) for t in thresholds]
    return np.mean(f_scores)
