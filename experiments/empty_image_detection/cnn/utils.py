import pandas as pd


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
