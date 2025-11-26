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


def get_2d_mask_from_rle(rle_string: str, shape=SHAPE) -> np.ndarray:
    """
    Convert an RLE string to a 2D binary mask.

    :param rle_string: Run-length encoded mask as a string
    :param shape: Tuple of (height, width) for the output mask
    :return: 2D numpy array of 0s and 1s representing the mask
    """
    if not rle_string or rle_string.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    rle = np.fromstring(rle_string, sep=" ", dtype=int)
    starts, lengths = rle[0::2] - 1, rle[1::2]
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask[s:e] = 1

    return mask.reshape(shape, order="F")


def get_rle_from_2d_mask(mask: np.ndarray) -> str:
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
        mask = get_2d_mask_from_rle(rle, shape)
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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    # skip label 0 (background)
    sizes = stats[1:, cv2.CC_STAT_AREA].tolist()
    return sizes


def keep_largest_shape(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)

    return largest_mask


def write_submission_csv(results, out_path="submission.csv") -> None:
    """
    Write Kaggle-format submission CSV.

    :param results: A list where each element is a tuple
        (image_id, list_of_rle_strings). If the list of RLE strings is
        empty, a single blank row is written for that image.
    :param out_path: The output CSV filename.
    :return: None
    """
    rows = []
    for image_id, rle_list in results:
        rle_list = rle_list if rle_list else [""]
        rows.extend({"ImageId": image_id, "EncodedPixels": rle} for rle in rle_list)

    df = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} ({len(df)} rows)")


def get_image(img_id: str) -> np.ndarray:
    """
    Load an image and its ground truth masks.

    :param img_id: Image filename
    :returns: image array
    """
    img = cv2.imread(f"{IMAGE_PATH}/{img_id}")

    if not img:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}/{img_id}")

    return img[:, :, ::-1]  # BGR→RGB


def iou(a: np.ndarray, b: np.ndarray) -> float:
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
            m
            for m in masks_sorted
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
        # Remove pixels already assigned
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


def rle_to_mask(rle : str | float | None, H : int, W: int) -> np.ndarray:
     # Handle None, NaN, or float('nan') from pandas
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return np.zeros((H, W), dtype=np.uint8)

    # Handle empty string masks
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros((H, W), dtype=np.uint8)
    
    mask = np.zeros(H * W, dtype=np.uint8)
    if rle.strip() == "":
        return mask.reshape((H, W))
    s = list(map(int, rle.split()))
    for start, length in zip(s[0::2], s[1::2]):
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape((H, W), order="F")


def rles_to_masks(rles: List[str], H: int, W: int) -> List[np.ndarray]:
    """
    Convert a list of RLE strings to a list of 2D binary masks.

    :param rles: List of RLE strings
    :param H: Height of the output masks
    :param W: Width of the output masks
    :return: List of 2D numpy arrays representing the masks
    """
    return [rle_to_mask(rle, H, W) for rle in rles]

def compute_confusion_counts(iou_mat: np.ndarray, t: np.float32) -> tuple[int, int, int]:
    match_matrix = (iou_mat >= t)
    
    # iou_mat shape is (g, p)
    gt_matched = match_matrix.any(axis=1)  # per GT mask
    pred_matched = match_matrix.any(axis=0)  # per predicted mask

    TP = gt_matched.sum()
    FN = (~gt_matched).sum() # each unmatched gt is a false negative
    FP = (~pred_matched).sum() # each unmatched prediction is a false positive

    return int(TP), int(FN), int(FP)

def fetch_image_from_dict(img_dict: Dict[str, Any]) -> Iterator[tuple[np.ndarray, list[str]]]:
    for img_id, rles in img_dict.items():
        img_t = cv2.imread(f"{IMAGE_PATH}/{img_id}")
        if img_t is None:
            raise FileNotFoundError(f"Image not found: {IMAGE_PATH}/{img_id}")
        img : np.ndarray = img_t[:, :, ::-1]  # BGR→RGB
        yield (img, rles)    


def f_score(t: np.float32, iou_mat: np.ndarray, beta: float = 2.0) -> float:
    TP, FN, FP = compute_confusion_counts(iou_mat, t)

    num = (1 + beta**2) * TP
    den = (1 + beta**2) * TP + (beta**2) * FN + FP

    return num / den if den > 0 else 0.0

def average_f_score_of_image(iou_mat: np.ndarray, beta: float = 2.0) -> np.float32:
    thresholds: NDArray[np.float32] = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    f_scores = [f_score(t, iou_mat, beta) for t in thresholds]
    return np.mean(f_scores)