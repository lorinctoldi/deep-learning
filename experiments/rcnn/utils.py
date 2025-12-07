import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple, Union


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    :param a: First binary mask as a NumPy array.
    :param b: Second binary mask as a NumPy array.
    :returns: IoU score as a float between 0 and 1.
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
            m
            for m in masks_sorted
            if iou(current["segmentation"], m["segmentation"]) <= iou_thresh
        ]

    return keep


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


def rle_to_mask(rle: Union[str, float, None], H: int, W: int) -> np.ndarray:
    """
    Convert a Run-Length Encoded (RLE) string to a binary mask.

    :param rle: RLE string, or None/NaN/empty if no mask is present.
    :param H: Height of the output mask.
    :param W: Width of the output mask.
    :returns: Binary mask of shape (H, W) as a NumPy uint8 array.
    """
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


def rles_to_masks(rles: List[str], H: int, W: int) -> List[np.ndarray]:
    """
    Convert a list of RLE strings to a list of 2D binary masks.

    :param rles: List of RLE strings
    :param H: Height of the output masks
    :param W: Width of the output masks
    :return: List of 2D numpy arrays representing the masks
    """
    return [rle_to_mask(rle, H, W) for rle in rles]

def compute_confusion_counts(iou_mat: np.ndarray, t: float) -> Tuple[int, int, int]:
    """
    Compute true positives, false negatives, and false positives from an IoU matrix.

    Each ground truth and predicted mask is considered matched if their IoU exceeds the threshold `t`.

    :param iou_mat: 2D NumPy array where element (i, j) is the IoU between ground truth mask i and predicted mask j.
    :param t: IoU threshold to consider a prediction as a true positive.
    :returns: Tuple of (TP, FN, FP) counts as integers.
    """
    match_matrix = (iou_mat >= t)
    
    gt_matched = match_matrix.any(axis=1)
    pred_matched = match_matrix.any(axis=0)

    TP = gt_matched.sum()
    FN = (~gt_matched).sum()
    FP = (~pred_matched).sum()

    return int(TP), int(FN), int(FP)


def f_score(t: float, iou_mat: np.ndarray, beta: float = 2.0) -> float:
    """
    Compute the F-score for a given IoU matrix and threshold.

    :param t: IoU threshold to consider a prediction as a true positive.
    :param iou_mat: 2D NumPy array where element (i, j) is the IoU between ground truth mask i and predicted mask j.
    :param beta: Weight of recall relative to precision in the F-score (default is 2.0).
    :returns: F-score as a float between 0 and 1.
    """
    TP, FN, FP = compute_confusion_counts(iou_mat, t)

    num = (1 + beta**2) * TP
    den = (1 + beta**2) * TP + (beta**2) * FN + FP

    return num / den if den > 0 else 0.0

def average_f_score_of_image(iou_mat: np.ndarray, beta: float = 2.0) -> float:
    """
    Compute the average F-score of an image across multiple IoU thresholds.

    :param iou_mat: 2D NumPy array where element (i, j) is the IoU between ground truth mask i and predicted mask j.
    :param beta: Weight of recall relative to precision in the F-score (default is 2.0).
    :returns: Average F-score across IoU thresholds from 0.5 to 0.95.
    """
    thresholds: NDArray[np.float32] = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    f_scores = [f_score(t, iou_mat, beta) for t in thresholds]
    return np.mean(f_scores)
