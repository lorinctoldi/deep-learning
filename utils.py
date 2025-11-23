import os
import pickle
import numpy as np
import pandas as pd
from collections import deque
from typing import Sequence, List, Tuple, Dict, Any
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
    return np.any(mask > 1) # type: ignore


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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # skip label 0 (background)
    sizes = stats[1:, cv2.CC_STAT_AREA].tolist()
    return sizes

def keep_largest_shape(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

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
        rows.extend(
            {"ImageId": image_id, "EncodedPixels": rle}
            for rle in rle_list
        )

    df = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} ({len(df)} rows)")

def get_image(img_id: str) -> np.ndarray:
    """
    Load an image and its ground truth masks.

    :param img_id: Image filename
    :returns: image array
    """
    img = cv2.imread(f"{IMAGE_PATH}/{img_id}")[:, :, ::-1]  # BGR→RGB
    
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img}")

    return img

def split_mask_into_blobs(mask):
    """
    Takes a SAM mask and returns a list of single-blob binary masks.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    blobs = []
    for label in range(1, num_labels):  # label 0 = background
        blob = (labels == label).astype(np.uint8)
        blobs.append(blob)

    return blobs

def filter_by_min_area(mask: np.ndarray, min_area_px: int = 12) -> bool:
    """
    Check if a single binary mask passes the area threshold for the 'area' role.

    :param mask: 2D binary mask
    :param min_area_px: Minimum area in pixels to keep a component
    :returns: False if mask passes the area rule, True otherwise
    """
    mask_area = mask.sum()

    return mask_area < min_area_px

def filter_by_max_ratio(mask: np.ndarray, max_ratio: float = 0.20) -> bool:
    """
    Check if a single binary mask passes the maximum area ratio rule.

    :param mask: 2D binary mask
    :param max_ratio: Maximum area as a fraction of total image area
    :returns: False if mask area is below the max_ratio, True otherwise
    """
    mask_area = mask.sum()
    H, W = mask.shape
    total_area = H * W

    return mask_area > (max_ratio * total_area)

def filter_by_rectangularity(mask: np.ndarray, min_rectangularity_ratio: float = 0.7) -> bool:
    """
    Check if a binary mask is approximately rectangular.

    :param mask: 2D binary mask
    :param min_rectangularity_ratio: Minimum ratio of mask area to bounding rectangle area
    :returns: False if mask is sufficiently rectangular, True otherwise
    """
    ys, xs = np.nonzero(mask)
    
    if len(xs) < 3 or len(ys) < 3:
        return True  # too small to evaluate shape

    # Coordinates of all points
    points = np.vstack((xs, ys)).T.astype(np.float32)

    # Minimum-area rectangle (rotated)
    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    rect_area = width * height

    if rect_area <= 0:
        return True

    mask_area = mask.sum()
    rectangularity_ratio = mask_area / rect_area

    return rectangularity_ratio < min_rectangularity_ratio

def filter_by_eccentricity(mask: np.ndarray, max_eccentricity: float = 0.995) -> bool:
    """
    Reject overly elongated masks using eccentricity computed from PCA/covariance.

    :returns: True if mask should be rejected, False otherwise
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return True  # too small to evaluate

    # PCA via covariance
    cov = np.cov(np.vstack([xs, ys]))
    eigvals, _ = np.linalg.eig(cov)

    # ratio of principal axes (0 = circle, 1 = line)
    ecc = 1 - (min(eigvals) / (max(eigvals) + 1e-9))

    return ecc > max_eccentricity


def filter_mask(mask: np.ndarray,
                min_area_px: int = 40,
                max_area_ratio: float = 0.12,
                min_rectangularity_ratio: float = 0.6) -> bool:
    """
    Apply all mask filters in a pipeline for a single mask.
    
    Calls:
        - filter_by_max_ratio
        - filter_by_min_area
        - filter_by_rectangularity
        - filter_by_eccentricity

    :param mask: 2D binary mask
    :returns: False if mask passes all filters, True otherwise
    """
    if filter_by_max_ratio(mask, max_ratio=max_area_ratio):
        return True
    
    if filter_by_min_area(mask, min_area_px=min_area_px):
        return True
    
    # if filter_by_rectangularity(mask, min_rectangularity_ratio=min_rectangularity_ratio):
    #     return True
    
    if filter_by_eccentricity(mask):
        return True
    
    return False

def intersection_over_union(a : np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0

def iou_validation(g : List[np.ndarray], p: List[np.ndarray]) -> np.ndarray:
    """
    Compute IoU matrix between ground truth masks and predicted masks.
    
    :param g: List of ground truth masks (2D numpy arrays)
    :param p: List of predicted masks (2D numpy arrays)
    :returns: IoU matrix of shape (len(g), len(p))
    """
    iou_matrix = np.zeros((len(g), len(p)), dtype=np.float32)
    for i, gm in enumerate(g):
        for j, pm in enumerate(p):
            iou_matrix[i, j] = intersection_over_union(gm, pm)
    return iou_matrix

def non_max_mask_suppression(masks, iou_thresh=0.8, score_key='predicted_iou'):
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
            if intersection_over_union(current['segmentation'], m['segmentation']) <= iou_thresh
        ]

    return keep

def enforce_no_overlap(masks: list[dict], score_key: str = 'predicted_iou') -> list[dict]:
    """
    Remove pixel-level overlaps between masks by giving pixels to the higher-scoring mask.
    
    :param masks: List of mask dicts with 'segmentation' keys
    :param score_key: Key in mask dicts used to sort by score
    :return: List of masks with no overlapping pixels
    """
    masks_sorted = sorted(masks, key=lambda m: m[score_key], reverse=True)
    
    occupied = np.zeros_like(masks_sorted[0]['segmentation'], dtype=bool)
    final_masks = []

    for mask in masks_sorted:
        seg = mask['segmentation'].copy()
        # Remove pixels already assigned
        seg[occupied] = False
        if seg.any():
            final_masks.append({**mask, 'segmentation': seg})
            occupied |= seg  # mark these pixels as taken

    return final_masks