import random

import numpy as np
import torch

from scripts.train_uiis_fixed_protocol import atomic_torch_save, checkpoint_payload, load_completed_epoch_checkpoint


def test_atomic_checkpoint_round_trip_restores_training_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    path = tmp_path / "last.pt"
    atomic_torch_save(checkpoint_payload(2, 17, model, optimizer, scheduler, scaler, {"seed": 5}), path)
    assert path.is_file()
    assert not path.with_suffix(".pt.tmp").exists()

    resumed_model = torch.nn.Linear(2, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=3)
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, global_step = load_completed_epoch_checkpoint(
        path, resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler, torch.device("cpu")
    )
    assert (start_epoch, global_step) == (3, 17)
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
