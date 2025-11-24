from typing import Any
import numpy as np
import cv2

from util.general_utils import iou

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
    
    # if filter_by_eccentricity(mask):
    #     return True
    
    return False

def non_max_mask_suppression(masks, iou_thresh=0.8, score_key='predicted_iou'):
    """
    Apply non-maximum suppression to masks based on IoU and a score key.
    This removes overlapping masks, keeping the highest-scoring ones.
    
    :param masks: List of mask dicts from SAM
    :param iou_thresh: IoU threshold above which to suppress masks
    :param score_key: Key in mask dicts to use for scoring
    """
    order = sorted(range(len(masks)), key=lambda i: masks[i][score_key], reverse=True)
    taken = []
    keep = []
    for i in order:
        if i in taken: 
            continue
        keep.append(masks[i])
        for j in order:
            if j in taken or j == i: 
                continue
            if iou(masks[i]['segmentation'], masks[j]['segmentation']) > iou_thresh:
                taken.append(j)
    return keep

def enforce_no_overlap(masks : list[dict[str, Any]], score_key : str ='predicted_iou'):
    if not masks:
        return []
    # If tiny overlaps remain, give pixels to the higher-score mask
    masks_sorted = sorted(masks, key=lambda m: m[score_key], reverse=True)
    acc = np.zeros_like(masks_sorted[0]['segmentation'], dtype=np.int16)
    final = []
    for m in masks_sorted:
        seg = m['segmentation'].copy()
        seg[acc > 0] = False
        if seg.sum() == 0:
            continue
        final.append({**m, 'segmentation': seg})
        acc[seg] = 1
    return final