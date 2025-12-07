import pandas as pd
import numpy as np
import cv2

from constants import IMAGE_PATH


def load_masks(filter_empty: bool = False, original=False) -> pd.DataFrame:
    """
    Load the segmentation RLEs from CSV.

    :param filter_empty: If True, remove rows where 'EncodedPixels' is empty
    :param original: If True, load from original_segmentations.csv
    :return: DataFrame with columns 'ImageId' and 'EncodedPixels'
    """
    path = (
        "../../../data/original_segmentations.csv" if original else "../../../data/segmentations.csv"
    )
    masks = pd.read_csv(path).fillna("")
    if filter_empty:
        masks = masks[masks.EncodedPixels != ""]
    return masks


def get_image(img_id: str) -> np.ndarray:
    """
    Load an image and its ground truth masks.

    :param img_id: Image filename
    :returns: image array
    """
    img = cv2.imread(f"{IMAGE_PATH}/{img_id}.jpg")

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}/{img_id}.jpg")

    return img[:, :, ::-1]
