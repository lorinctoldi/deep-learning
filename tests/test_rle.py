import numpy as np

from utils.general_utils import get_mask_from_rle  # <-- adjust import


def test_empty_rle_returns_zero_mask():
    H, W = 4, 5
    rle = ""
    mask = get_mask_from_rle(rle, (H, W))

    assert mask.shape == (H, W)
    assert np.all(mask == 0)


def test_single_run():
    # Simple 3x3 mask, turn on positions 2–4 in column-major indexing
    # Column-major layout: positions are filled DOWN each column first.
    #
    # Flat indices (1-indexed):
    # col0: 1,2,3
    # col1: 4,5,6
    # col2: 7,8,9
    #
    # Positions 2,3,4 should be 1.
    H, W = 3, 3
    rle = "2 3"  # start=2, length=3
    mask = get_mask_from_rle(rle, (H, W))

    expected = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )

    assert mask.shape == (H, W)
    assert np.array_equal(mask, expected)


def test_multiple_runs():
    # 4x3 mask
    # Column-major positions:
    # col0: 1,2,3,4
    # col1: 5,6,7,8
    # col2: 9,10,11,12
    #
    # Runs:
    #   2 2  → positions 2,3
    #   7 1  → position 7
    #   10 2 → positions 10,11
    H, W = 4, 3
    rle = "2 2 7 1 10 2"
    mask = get_mask_from_rle(rle, (H, W))

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(mask, expected)


def test_column_major_correctness():
    # This explicitly ensures that order='F' reshape is respected.
    H, W = 3, 2
    #
    # Column-major indexing:
    # col0: 1,2,3
    # col1: 4,5,6
    #
    # Activate pixels 1 and 4 → should be (0,0) and (0,1)
    rle = "1 1 4 1"
    mask = get_mask_from_rle(rle, (H, W))

    expected = np.array(
        [
            [1, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(mask, expected)
