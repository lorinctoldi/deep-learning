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
    img = cv2.imread(img_path)[:, :, ::-1]      # type: ignore # BGR→RGB
    
    cv2.imwrite("data/cache/sam_debug.png", img[:, :, ::-1])  # type: ignore # RGB→BGR
    
    masks = []
    
    MASK_FILE_PATH = "data/cache/sam_masks.pkl"

    if os.path.exists(MASK_FILE_PATH):
        masks = joblib.load(MASK_FILE_PATH)
    else:
        masks = mask_generator.generate(img)        # list of dicts
        joblib.dump(masks, MASK_FILE_PATH)
    return img, masks


# Remove obvious junk (cloud, shorlenine bits)
def rule_filter(masks, H, W,
                min_area_px=40,
                max_area_ratio=0.12,
                max_eccentricity=0.995):
    keep = []
    img_area = H * W
    for m in masks:
        seg = m['segmentation']
        area = seg.sum()
        if area < min_area_px: 
            continue
        if area > max_area_ratio * img_area:
            continue
        # optional: eccentricity via second moments (proxy)
        ys, xs = np.where(seg)
        if len(xs) < 3:
            continue
        cov = np.cov(np.vstack([xs, ys]))
        eigvals, _ = np.linalg.eig(cov)
        ecc = 1 - (min(eigvals) / max(eigvals) + 1e-9)
        if ecc > max_eccentricity:
            continue
        keep.append(m)
    return keep

# intersection over union
def iou(a, b):
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