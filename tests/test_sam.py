import numpy as np
from unittest.mock import MagicMock
from sam_predictor import get_image_iou_mat
import pytest

@pytest.fixture
def dummy_img():
    # A simple 5x5 RGB image (all zeros)
    return np.zeros((5, 5, 3), dtype=np.uint8)


@pytest.fixture
def dummy_masks():
    # Pretend SAM found exactly 1 mask covering whole image
    return [np.ones((5, 5), dtype=bool)]


@pytest.fixture
def dummy_gt_mask():
    # Ground truth also full mask
    return np.ones((5, 5), dtype=bool)


def test_evaluate_image_with_valid_rle(monkeypatch, dummy_img, dummy_masks, dummy_gt_mask):
    mock_pipeline = MagicMock(return_value=dummy_masks)
    monkeypatch.setattr("sam_predictor.mask_generator_pipeline", mock_pipeline)

    expected = np.array([[0.5]], dtype=np.float32)
    mock_iou_matrix = MagicMock(return_value=expected)
    monkeypatch.setattr("sam_predictor.compute_iou_matrix", mock_iou_matrix)

    img_data = (dummy_img, ["1 5 20 10"])
    result = get_image_iou_mat(img_data)

    mock_pipeline.assert_called_once_with(dummy_img)
    mock_iou_matrix.assert_called_once()
    np.testing.assert_array_equal(result, expected)


def test_evaluate_image_with_empty_rle(monkeypatch, dummy_img, dummy_masks):
    mock_pipeline = MagicMock(return_value=dummy_masks)
    monkeypatch.setattr("sam_predictor.mask_generator_pipeline", mock_pipeline)

    expected = np.array([[0.0]], dtype=np.float32)
    mock_iou_matrix = MagicMock(return_value=expected)
    monkeypatch.setattr("sam_predictor.compute_iou_matrix", mock_iou_matrix)

    img_data = (dummy_img, [""])
    result = get_image_iou_mat(img_data)

    mock_pipeline.assert_called_once_with(dummy_img)
    mock_iou_matrix.assert_called_once()
    np.testing.assert_array_equal(result, expected)
