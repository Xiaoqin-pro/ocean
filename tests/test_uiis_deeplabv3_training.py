import random

import numpy as np
import torch

from scripts.train_uiis_deeplabv3_fixed_protocol import atomic_torch_save, build_model, checkpoint_payload, load_completed_epoch_checkpoint, resize_logits


def test_uiis_deeplabv3_shape_without_downloading_weights():
    model = build_model(8, None).eval()
    with torch.no_grad():
        logits = model(torch.zeros((1, 3, 64, 64)))["out"]
    assert resize_logits(logits, torch.zeros((1, 64, 64), dtype=torch.long)).shape == (1, 8, 64, 64)


def test_uiis_deeplabv3_atomic_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    path = tmp_path / "last.pt"
    atomic_torch_save(checkpoint_payload(2, 19, model, optimizer, scheduler, scaler, {"seed": 9}), path)
    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert load_completed_epoch_checkpoint(path, restored_model, restored_optimizer, restored_scheduler, restored_scaler, torch.device("cpu")) == (3, 19)
    assert torch.equal(model.weight, restored_model.weight)
