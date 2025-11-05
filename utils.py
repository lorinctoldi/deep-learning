import os
import pickle
import numpy as np
import pandas as pd
from collections import deque
from typing import Sequence, List, Tuple
from PIL import Image
from IPython.display import display

from constants import SHAPE, CACHE_PATH


def load_masks(filter_empty: bool = False, original=False) -> pd.DataFrame:
    """
    Load the segmentation RLEs from CSV.

    :param filter_empty: If True, remove rows where 'EncodedPixels' is empty
    :param original: If True, load from original_segmentations.csv
    :return: DataFrame with columns 'ImageId' and 'EncodedPixels'
    """
    path = (
        "./data/original_segmentations.csv" if original else "./data/segmentations.csv"
    )
    masks = pd.read_csv(path).fillna("")
    if filter_empty:
        masks = masks[masks.EncodedPixels != ""]
    return masks


def get_2d_mask_from_rle(rle_string: str, shape=SHAPE) -> np.ndarray:
    """
    Convert an RLE string to a 2D binary mask.

    :param rle_string: Run-length encoded mask as a string
    :param shape: Tuple of (height, width) for the output mask
    :return: 2D numpy array of 0s and 1s representing the mask
    """
    rle = np.fromstring(rle_string, sep=" ", dtype=int)
    starts, lengths = rle[0::2] - 1, rle[1::2]
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask[s:e] = 1

    return mask.reshape(shape, order="F")


def get_rle_from_2d_mask(mask: np.ndarray) -> str:
    """
    Convert a 2D binary mask to a run-length encoded (RLE) string.

    :param mask: 2D numpy array of 0s and 1s
    :return: RLE string (empty string if mask has no positive pixels)
    """
    pixels = mask.flatten(order="F")

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    starts = runs[::2]
    ends = runs[1::2]
    lengths = ends - starts

    rle = " ".join([f"{s} {l}" for s, l in zip(starts, lengths)])
    return rle


def rles_to_combined_mask(rle_list, shape=SHAPE):
    """
    Convert a list of RLE strings into a single 2D mask array.

    :param rle_list: List of RLE strings (one per boat)
    :param shape: Tuple of (height, width)
    :return: 2D numpy array with summed mask values; pixels >1 indicate overlap
    """
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        mask = get_2d_mask_from_rle(rle, shape)
        combined_mask += mask
    return combined_mask


def has_overlap(mask) -> bool:
    """
    Check whether a 2D mask contains overlapping shapes.

    In a combined mask, overlapping areas (where multiple shapes occupy
    the same pixels) will have values greater than 1.

    :param mask: 2D numpy array representing a combined mask
    :return: True if any pixel has a value >1, indicating overlap; False otherwise
    """
    return np.any(mask > 1) # type: ignore


def use_cache(func):
    def wrapper(*args, **kwargs):
        cache_file = os.path.join(CACHE_PATH, f"{func.__name__}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    return wrapper


def visualize_image_with_mask(image_id: str, mask: np.ndarray) -> None:
    """
    Visualize an image with its mask overlaid.

    :param image_id: File name of the image
    :param mask: 2D numpy array of 0s and 1s representing the mask
    """
    img = Image.open(f"./data/images/{image_id}").convert("RGBA")

    if mask.shape != (img.height, img.width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {(img.height, img.width)}"
        )

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    red_overlay = Image.new("RGBA", img.size, color=(255, 0, 0, 100))
    mask_overlay = Image.composite(red_overlay, Image.new("RGBA", img.size), mask_img)

    combined = Image.alpha_composite(img, mask_overlay)

    display(combined)


def get_number_of_shapes_and_sizes(grid: Sequence[Sequence[int]]) -> List[int]:
    """
    Compute the sizes (number of pixels) of all connected shapes in a 2D binary grid.

    A shape is defined as a group of adjacent 1s in the grid.
    In this implementation, pixels that touch horizontally, vertically, or diagonally
    are considered connected and belong to the same shape.

    Inspired by the "Counting Islands" problem on LeetCode:
    https://leetcode.com/problems/number-of-islands/

    :param grid: A 2D grid of 0s and 1s representing a binary mask
    :return: List[int]: A list containing the size (number of pixels) of each shape
    found in the grid
    """
    visited = set()
    rows, cols = len(grid), len(grid[0])
    sizes = []

    def bfs(r: int, c: int) -> int:
        """
        Breadth-first search to traverse all connected pixels starting from (r, c).
        Counts the number of pixels in the current shape.
        """
        q = deque()
        visited.add((r, c))
        q.append((r, c))
        size = 0

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        while q:
            row, col = q.popleft()
            size += 1
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and grid[nr][nc] == 1
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return size

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                sizes.append(bfs(r, c))

    return sizes


def keep_largest_shape(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected shape in a 2D binary mask.
    All smaller shapes are removed.

    :param mask: 2D numpy array of 0s and 1s
    :return: new 2D mask with only the largest shape
    """
    visited = set()
    rows, cols = mask.shape
    largest_shape_size = 0
    largest_shape_coords: List[Tuple[int, int]] = []

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def bfs(r: int, c: int) -> List[Tuple[int, int]]:
        q = deque()
        q.append((r, c))
        visited.add((r, c))
        shape_coords = [(r, c)]

        while q:
            row, col = q.popleft()
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and mask[nr, nc] == 1
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    shape_coords.append((nr, nc))
        return shape_coords

    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 1 and (r, c) not in visited:
                coords = bfs(r, c)
                if len(coords) > largest_shape_size:
                    largest_shape_size = len(coords)
                    largest_shape_coords = coords

    new_mask = np.zeros_like(mask)
    for r, c in largest_shape_coords:
        new_mask[r, c] = 1

    return new_mask


import pandas as pd

def write_submission_csv(results, out_path="submission.csv"):
    """
    Write Kaggle-format submission CSV.

    Parameters
    ----------
    results : list of tuples
        Each element = (image_id, list_of_rle_strings)
        If list_of_rle_strings is empty, one blank row is written.
    out_path : str
        Output CSV filename.
    """
    rows = []
    for image_id, rles in results:
        if not rles:                           # no ships
            rows.append({"ImageId": image_id, "EncodedPixels": ""})
        else:
            for rle in rles:                   # one row per ship
                rows.append({"ImageId": image_id, "EncodedPixels": rle})

    df = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} ({len(df)} rows)")
