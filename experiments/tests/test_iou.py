import numpy as np
from unittest.mock import MagicMock

from util.general_utils import compute_iou_matrix


def test_compute_iou_matrix_basic():
    # Two identical masks and one empty
    g = [np.array([[1, 0], [0, 1]], dtype=bool)]
    p = [
        np.array([[1, 0], [0, 1]], dtype=bool),  # exact same → IoU = 1
        np.zeros((2, 2), dtype=bool)              # no overlap → IoU = 0
    ]

    result = compute_iou_matrix(g, p)
    assert result.shape == (1, 2)
    np.testing.assert_array_almost_equal(result, np.array([[1.0, 0.0]], dtype=np.float32))


def test_compute_iou_matrix_no_ground_truth():
    g = []
    p = [np.ones((3, 3), dtype=bool)]

    result = compute_iou_matrix(g, p)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 1)  # zero rows, one column


def test_compute_iou_matrix_no_predictions():
    g = [np.ones((3, 3), dtype=bool)]
    p = []

    result = compute_iou_matrix(g, p)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 0)  # one row, zero columns


def test_compute_iou_matrix_both_empty():
    g = []
    p = []

    result = compute_iou_matrix(g, p)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    assert result.shape == (0, 0)


def test_compute_iou_matrix_calls_iou(monkeypatch):
    # Four masks → IoU called 4 times (2 GT x 2 predictions)
    g = [np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]
    p = [np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]

    mock_iou = MagicMock(return_value=0.5)
    monkeypatch.setattr("util.general_utils.iou", mock_iou)

    result = compute_iou_matrix(g, p)

    assert result.shape == (2, 2)
    mock_iou.assert_called()
    assert mock_iou.call_count == len(g) * len(p)
    assert np.all(result == 0.5)