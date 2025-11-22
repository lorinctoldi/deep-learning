import numpy as np
import pytest
from unittest.mock import MagicMock

from sam_predictor import evaluate_image


@pytest.fixture
def dummy_img():
    return np.zeros((5, 5, 3), dtype=np.uint8)


@pytest.fixture
def dummy_masks():
    return [np.ones((5, 5), dtype=bool)]  # pretend SAM prediction


@pytest.fixture
def dummy_gt_mask():
    return np.ones((5, 5), dtype=bool)


def test_evaluate_image_with_valid_rle(monkeypatch, dummy_img, dummy_masks, dummy_gt_mask):
    # mock mask_generator_pipeline
    mock_pipeline = MagicMock(return_value=dummy_masks)
    monkeypatch.setattr("sam_predictor.mask_generator_pipeline", mock_pipeline)

    # mock rle_to_mask
    mock_rle_to_mask = MagicMock(return_value=dummy_gt_mask)
    monkeypatch.setattr("sam_predictor.rle_to_mask", mock_rle_to_mask)

    # mock compute_iou_matrix
    expected = np.array([[0.5]], dtype=np.float32)
    mock_iou = MagicMock(return_value=expected)
    monkeypatch.setattr("sam_predictor.compute_iou_matrix", mock_iou)

    img_data = (dummy_img, "1 5 20 10")
    result = evaluate_image(img_data)

    mock_pipeline.assert_called_once_with(dummy_img)
    mock_rle_to_mask.assert_called_once_with("1 5 20 10", 5, 5)
    mock_iou.assert_called_once()

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1)
    np.testing.assert_array_equal(result, expected)


def test_evaluate_image_with_empty_rle(monkeypatch, dummy_img, dummy_masks):
    mock_pipeline = MagicMock(return_value=dummy_masks)
    monkeypatch.setattr("sam_predictor.mask_generator_pipeline", mock_pipeline)

    mock_rle_to_mask = MagicMock()
    monkeypatch.setattr("sam_predictor.rle_to_mask", mock_rle_to_mask)

    expected = np.array([[0.0]], dtype=np.float32)
    mock_iou = MagicMock(return_value=expected)
    monkeypatch.setattr("sam_predictor.compute_iou_matrix", mock_iou)

    img_data = (dummy_img, "   ")
    result = evaluate_image(img_data)

    mock_pipeline.assert_called_once_with(dummy_img)
    mock_rle_to_mask.assert_not_called()  # RLE was empty -> skip GT mask build
    mock_iou.assert_called_once_with([], dummy_masks)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)

