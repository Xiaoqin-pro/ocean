import math

import torch

from metrics.segmentation import confusion_matrix, segmentation_metrics


def test_hand_calculated_metrics() -> None:
    prediction = torch.tensor([[0, 0], [1, 1]])
    target = torch.tensor([[0, 1], [1, 1]])
    matrix = confusion_matrix(prediction, target, num_classes=2)
    assert torch.equal(matrix, torch.tensor([[1, 0], [1, 2]]))
    result = segmentation_metrics(prediction, target, num_classes=2)
    assert result["pixel_accuracy"] == 0.75
    assert math.isclose(result["per_class_iou"][0], 0.5)
    assert math.isclose(result["per_class_iou"][1], 2 / 3)
    assert math.isclose(result["miou"], 7 / 12)


def test_ignore_index_and_empty_class() -> None:
    prediction = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 255, 1])
    result = segmentation_metrics(prediction, target, num_classes=3)
    assert result["pixel_accuracy"] == 1.0
    assert math.isnan(result["per_class_iou"][2])
    assert result["miou"] == 1.0
