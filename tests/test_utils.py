import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils import (
    load_masks,
    get_mask_from_rle,
    get_rle_from_mask,
    rles_to_combined_mask,
    has_overlap,
    visualize_image_with_mask,
    get_number_of_shapes_and_sizes,
    keep_largest_shape,
    get_image,
    SHAPE,
    IMAGE_PATH,
)

# -------------------
# load_masks
# -------------------
def test_load_masks_returns_dataframe():
    df = load_masks()
    assert isinstance(df, pd.DataFrame)
    assert 'ImageId' in df.columns
    assert 'EncodedPixels' in df.columns

def test_load_masks_filter_empty():
    df = load_masks(filter_empty=True)
    assert df['EncodedPixels'].eq('').sum() == 0

# -------------------
# get_mask_from_rle
# -------------------
def test_get_mask_from_rle_basic():
    mask = get_mask_from_rle("1 3 10 2", shape=(5, 5))
    assert mask.shape == (5, 5)
    assert np.sum(mask) == 5  # 3 + 2 pixels set

def test_get_mask_from_rle_empty_and_none():
    mask_none = get_mask_from_rle(None)
    mask_nan = get_mask_from_rle(float('nan'))
    mask_empty = get_mask_from_rle("")
    assert np.all(mask_none == 0)
    assert np.all(mask_nan == 0)
    assert np.all(mask_empty == 0)

# -------------------
# get_rle_from_mask
# -------------------
def test_get_rle_from_mask_basic():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 1] = 1
    rle = get_rle_from_mask(mask)
    assert isinstance(rle, str)
    mask_reconstructed = get_mask_from_rle(rle, shape=(5,5))
    assert np.array_equal(mask, mask_reconstructed)

def test_get_rle_from_mask_empty():
    mask = np.zeros((5, 5), dtype=np.uint8)
    rle = get_rle_from_mask(mask)
    assert rle == ""

# -------------------
# rles_to_combined_mask
# -------------------
def test_rles_to_combined_mask_combines_masks():
    rle1 = "1 2"
    rle2 = "3 2"
    combined = rles_to_combined_mask([rle1, rle2], shape=(3,3))
    assert combined.shape == (3,3)
    assert np.sum(combined) == 4

# -------------------
# has_overlap
# -------------------
def test_has_overlap_detects_overlap():
    mask = np.array([[1, 0], [0, 2]])
    assert bool(has_overlap(mask)) is True
    mask = np.array([[1, 0], [0, 1]])
    assert bool(has_overlap(mask)) is False

# -------------------
# get_number_of_shapes_and_sizes
# -------------------
def test_get_number_of_shapes_and_sizes_counts_shapes():
    mask = np.array([
        [1,0,0],
        [1,0,1],
        [0,0,1]
    ])
    sizes = get_number_of_shapes_and_sizes(mask)
    assert sorted(sizes) == [2, 2]

# -------------------
# keep_largest_shape
# -------------------
def test_keep_largest_shape_returns_largest():
    mask = np.array([
        [1,0,0],
        [1,0,1],
        [0,0,1]
    ])
    largest = keep_largest_shape(mask)
    assert largest.shape == mask.shape
    assert np.sum(largest) == max(get_number_of_shapes_and_sizes(mask))
