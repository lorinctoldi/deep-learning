from pathlib import Path
from typing import Callable, Dict, Iterator, List, Any
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import numpy as np
import joblib
import os
import cv2
from tqdm import tqdm

from constants import IMAGE_PATH, DataSplit
from util.general_utils import average_f_score_of_image, compute_iou_matrix, enforce_no_overlap, fetch_image_from_dict, iou, non_max_mask_suppression, rle_to_mask, rles_to_masks
from util.sam_utils import filter_mask
from numpy.typing import NDArray


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
        
def get_image_iou_mat(img_data : tuple[np.ndarray, list[str]], p : list[np.ndarray]) -> np.ndarray:
    img, rles = img_data

    g = rles_to_masks(rles, img.shape[0], img.shape[1])

    return compute_iou_matrix(g, p)
    

def evaluate_model(segmentation_pipeline : Callable[[NDArray[np.float32]], list[NDArray[np.float32]]]) -> np.float32:
    df = pd.read_csv("./data/segmentations.csv")

    rle_dict = (
        df.groupby("ImageId")["EncodedPixels"]
        .apply(list)
        .to_dict()
    )

    im_ids = list(rle_dict.keys())
    
    rng = np.random.default_rng(42)

    test_im_ids = rng.permutation(im_ids)[:100]


    print(f"Number of test images: {len(test_im_ids)}")

    test_rle_dict = {k: rle_dict[k] for k in test_im_ids}
    
    im_f2s : NDArray[np.float32] = np.array([], dtype=np.float32)

    for img, rles in tqdm(fetch_image_from_dict(test_rle_dict), total=len(test_rle_dict)):
        predicted = segmentation_pipeline(img)
        iou_mat : NDArray[np.float32] = get_image_iou_mat((img, rles), predicted)
        f2_score : np.float32 = average_f_score_of_image(iou_mat)
        im_f2s = np.append(im_f2s, f2_score)
        
    return np.mean(im_f2s)

def mask_generator_pipeline_sam(img : np.ndarray) -> List[np.ndarray]:
    """
    Given an image, generate masks using SAM and apply filtering.
    
    :param img: Input image as a numpy array
    :returns: List of filtered mask segmentations
    """
 
    masks : List[Dict[str, Any]] = mask_generator.generate(img)
    mask_filter = [filter_mask(m["segmentation"]) for m in masks]
    
    masks = [m for m, keep in zip(masks, mask_filter) if keep]
    
    masks = non_max_mask_suppression(masks, 0.8)
    masks = enforce_no_overlap(masks)
    return [np.asfortranarray(m['segmentation']) for m in masks]