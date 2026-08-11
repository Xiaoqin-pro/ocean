import torch

from scripts.train_deeplabv3_mobilenetv3 import build_model, resize_logits


def test_deeplabv3_mobilenetv3_smoke_shape_without_download():
    model = build_model(num_classes=8, backbone_weights=None).eval()
    with torch.no_grad():
        logits = model(torch.zeros((1, 3, 64, 64)))["out"]
    labels = torch.zeros((1, 64, 64), dtype=torch.long)
    assert resize_logits(logits, labels).shape == (1, 8, 64, 64)
