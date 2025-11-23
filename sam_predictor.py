from pathlib import Path
from typing import Dict, Generator, Iterator, List, Any
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import numpy as np
import joblib
import os
import cv2

from sympy import Tuple

from constants import IMAGE_PATH, DataSplit
from util.general_utils import compute_iou_matrix, iou, rle_to_mask, rles_to_masks

DEVICE = "cuda"

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,       # start modest; increase if recall is low
    pred_iou_thresh=0.88,     # SAM’s own confidence (tune later)
    stability_score_thresh=0.92,
    crop_n_layers=0,          # 0 to keep it fast at first
    output_mode="binary_mask"
)


def sam_candidates_bgr_path(img_path: str) -> tuple[np.ndarray, List[Dict[str, Any]]]:
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

def enforce_no_overlap(masks : List[Dict[str, Any]], score_key : str ='predicted_iou'):
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

def predict_masks(img_path: str):
    img, cand = sam_candidates_bgr_path(img_path)
    H, W = img.shape[:2]
    cand = rule_filter(cand, H, W)
    cand = non_max_mask_suppression(cand, 0.8)
    cand = enforce_no_overlap(cand)
    return [m['segmentation'] for m in cand]

def fetch_image(img_csv_path: DataSplit) -> Iterator[tuple[np.ndarray, str]]:
    scv_path = Path(img_csv_path.value)
    test_df = pd.read_csv(scv_path)
    for t in test_df.itertuples():
        img_t = cv2.imread(f"{IMAGE_PATH}/{t.ImageId}")
        if img_t is None:
            raise FileNotFoundError(f"Image not found: {IMAGE_PATH}/{t.ImageId}")
        img : np.ndarray = img_t[:, :, ::-1]  # BGR→RGB
        encoded = str(t.EncodedPixels) if pd.notna(t.EncodedPixels) else ""
        img_data = (img, encoded)
        yield img_data
        
def get_image_iou_mat(img_data : tuple[np.ndarray, list[str]]) -> np.ndarray:
    img, rles = img_data

    p : List[np.ndarray] = mask_generator_pipeline(img)

    g = rles_to_masks(rles, img.shape[0], img.shape[1])

    return compute_iou_matrix(g, p)
    

def evaluate_model(test_csv_path : str, img_root: str):
    df = pd.read_csv(test_csv_path)
    
    df.groupby('ImageId')
    
    
            
    # todo : continue evaluation pipeline

def mask_generator_pipeline(img : np.ndarray) -> List[np.ndarray]:
    """
    Given an image, generate masks using SAM and apply filtering.
    
    :param img: Input image as a numpy array
    :returns: List of filtered mask segmentations
    """
    H, W = img.shape[:2]
    masks : List[Dict[str, Any]] = mask_generator.generate(img)
    masks = rule_filter(masks, H, W)
    masks = non_max_mask_suppression(masks, 0.8)
    masks = enforce_no_overlap(masks)
    return [np.asfortranarray(m['segmentation']) for m in masks]