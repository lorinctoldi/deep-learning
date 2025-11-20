from typing import Dict, List, Any
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import numpy as np
import os
import cv2
import joblib

DEVICE = "cuda"

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,       # start modest; increase if recall is low
    pred_iou_thresh=0.88,     # SAM’s own confidence (tune later)
    stability_score_thresh=0.92,
    crop_n_layers=0,          # 0 to keep it fast at first
)


def sam_candidates_bgr_path(img_path: str):
    """
    Given an image path, use SAM to generate masks.
    
    :param img_path: Path to the input image
    :returns: Tuple of (image as numpy array, list of mask dicts)
    """
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    img = img[:, :, ::-1] # BGR→RGB
    
    cv2.imwrite("data/cache/sam_debug.png", img[:, :, ::-1])  # RGB→BGR
    
    masks = []
    
    MASK_FILE_PATH = "data/cache/sam_masks.pkl"

    if os.path.exists(MASK_FILE_PATH):
        masks = joblib.load(MASK_FILE_PATH)
    else:
        masks = mask_generator.generate(img)        # list of dicts
        joblib.dump(masks, MASK_FILE_PATH)
    return img, masks



def rule_filter(masks: List[Dict[str, Any]], H, W,
                min_area_px=40,
                max_area_ratio=0.12,
                max_eccentricity=0.995):
    """
    Apply heuristic rules to filter out unlikely masks, such as too big masks,
    too small masks, or overly elongated masks.
    
    :param masks: List of mask dicts from SAM
    :param H: Image height
    :param W: Image width
    :param min_area_px: Minimum area in percentage to full area of image to keep a mask
    :param max_area_ratio: Maximum area in percentage to full area of image to keep a mask
    :param max_eccentricity: Maximum eccentricity (0 to 1) to keep a mask
    :returns: Filtered list of mask dicts
    """
    keep : List[Dict[str, Any]] = []
    img_area = H * W
    for m in masks:
        seg : np.ndarray = m['segmentation']
        area = seg.sum()
        if area < min_area_px: 
            continue
        if area > max_area_ratio * img_area:
            continue
        
        ys, xs = np.nonzero(seg)
        if len(xs) < 3: # throw out very small mask blobs
            continue
        
        # we calculate the ellipsis it fits
        # then if it is too stretched, we throw it out
        cov = np.cov(np.vstack([xs, ys]))
        eigvals, _ = np.linalg.eig(cov)
        ecc = 1 - (min(eigvals) / max(eigvals) + 1e-9)
        if ecc > max_eccentricity:
            continue
        keep.append(m)
    return keep

# intersection over union
def iou(a : np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0

def non_max_mask_suppression(masks, iou_thresh=0.8, score_key='predicted_iou'):
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

def enforce_no_overlap(masks, score_key='predicted_iou'):
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

def predict_masks(img_path: str):
    img, cand = sam_candidates_bgr_path(img_path)
    H, W = img.shape[:2]
    cand = rule_filter(cand, H, W)
    cand = non_max_mask_suppression(cand, 0.8)
    cand = enforce_no_overlap(cand)
    return [m['segmentation'] for m in cand]

def rle_to_mask(rle, H, W):
    mask = np.zeros(H*W, dtype=np.uint8)
    if rle.strip() == "":
        return mask.reshape((H, W))
    s = list(map(int, rle.split()))
    for start, length in zip(s[0::2], s[1::2]):
        mask[start-1:start-1+length] = 1
    return mask.reshape((H, W), order='F')

def evaluate_model(test_csv_path : str, img_root: str):
    gt_data : Dict[str, str] = {}
    
    with open(test_csv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            img_id = parts[0]
            rle = parts[1]
            gt_data[img_id] = rle
            
    # todo : continue evaluation pipeline

def compute_iou_matrix(g : List[np.ndarray], p: List[np.ndarray]) -> np.ndarray:
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