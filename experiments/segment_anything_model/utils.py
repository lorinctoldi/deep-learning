import cv2
import numpy as np

from typing import List, Dict, Iterator, Any, Union, Tuple
from matplotlib import pyplot as plt
from constants import IMAGE_PATH


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
        return True

    points = np.vstack((xs, ys)).T.astype(np.float32)

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
        return True

    cov = np.cov(np.vstack([xs, ys]))
    eigvals, _ = np.linalg.eig(cov)

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
    
    if filter_by_rectangularity(mask, min_rectangularity_ratio=min_rectangularity_ratio):
        return True
    
    if filter_by_eccentricity(mask):
        return True
    
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


def enforce_no_overlap(
    masks: List[Dict[str, Any]],
    score_key: str = 'predicted_iou'
) -> List[Dict[str, Any]]:
    """
    Remove overlaps between masks by keeping higher-scoring masks first.

    :param masks: List of dictionaries, each containing at least a 'segmentation' (numpy array) and a score key.
    :param score_key: Key in each mask dict used to sort masks by confidence.
    :returns: List of masks with no overlapping pixels, higher-scoring masks retained.
    """
    if not masks:
        return []
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


def fetch_image_from_dict(img_dict: Dict[str, Any]) -> Iterator[Tuple[np.ndarray, List[str]]]:
    """
    Yield images and their associated RLE masks from a dictionary of image IDs.

    :param img_dict: Dictionary mapping image filenames to lists of RLE-encoded masks.
    :yields: Tuple of (image as an RGB NumPy array, list of RLE strings).
    :raises FileNotFoundError: If an image file cannot be found at the expected path.
    """
    for img_id, rles in img_dict.items():
        img_t = cv2.imread(f"{IMAGE_PATH}/{img_id}")
        if img_t is None:
            raise FileNotFoundError(f"Image not found: {IMAGE_PATH}/{img_id}")
        img : np.ndarray = img_t[:, :, ::-1]
        yield (img, rles)    


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
    thresholds: np.ndarray[np.float32] = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    f_scores = [f_score(t, iou_mat, beta) for t in thresholds]
    return np.mean(f_scores)


def get_image_iou_mat(
    img_data: Tuple[np.ndarray, List[str]],
    p: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the IoU matrix between ground truth masks and predicted masks for a single image.

    :param img_data: Tuple containing the image as a NumPy array and a list of RLE strings for ground truth masks.
    :param p: List of predicted masks as NumPy arrays (binary masks).
    :returns: 2D NumPy array where element (i, j) is the IoU between ground truth mask i and predicted mask j.
    """
    img, rles = img_data

    g = rles_to_masks(rles, img.shape[0], img.shape[1])

    return compute_iou_matrix(g, p)


def show_masks(
    img: np.ndarray,
    masks_list: List[Union[np.ndarray, Dict[str, Any]]]
) -> None:
    """
    Display an image with overlaid masks.

    :param img: Image as a NumPy array (H, W, 3).
    :param masks_list: List of masks to overlay. Each mask can be a binary NumPy array or a dict containing a 'segmentation' key.
    :returns: None. Displays the image using matplotlib.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    for m in masks_list:
        if isinstance(m, dict):
            seg = m['segmentation']
            plt.imshow(np.ma.masked_where(~seg, seg), alpha=0.45)
        else:
            plt.imshow(m, alpha=0.45)

    plt.show()
